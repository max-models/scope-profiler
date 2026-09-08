/* The optional HDF5 backend for the C region API; see scope_profiler.h.
 *
 * Compiled in only when SP_USE_HDF5 is defined, and included from
 * scope_profiler_impl.h so a build without it keeps its "libc and nothing
 * else" dependency set. When it is in, sp_profiler_finalize() writes
 * "<prefix>_rank<NNNNN>.h5" directly, in the same schema-2 layout
 * scope_profiler/h5writer.py produces -- so `scope-profiler inspect`, `plot`
 * and read_h5() open a C run's output with no import step.
 *
 * The layout written here is the one-rank case of that schema, and the two
 * files must change together:
 *
 *     /                       attrs: scope_profiler_schema = 2
 *                                    storage_layout = "columnar"
 *     /metadata               attrs only: the run's environment
 *     /region_table/names     [R] variable-length UTF-8, region name per id
 *     /rank_region_index/     [R] one row per (rank, region) pair
 *         region_ids          uint32   id into region_table/names
 *         ranks               uint32   this process's rank, on every row
 *         event_offsets       uint64   where the row's events start
 *         event_counts        uint64   how many it has
 *         source_lines        int64    -1 where unknown
 *         exclusive_totals    int64    -1: "not computed", as for an import
 *         source_files        vlen str "" where unknown
 *         source_texts        vlen str always "" -- no source is read here
 *         tags                vlen str always "[]" -- the C API has no tags
 *         summary_statistics  compound fixed-size stats, for summary readers
 *     /events/                [N] every event of every region, back to back
 *         start_times         int64    nanoseconds
 *         end_times           int64    nanoseconds
 *
 * The optional per-call columns of that schema (call_ids, parent_ids,
 * gpu_durations, the thread/task lanes) are *absent*, which is how the reader
 * is told this run did not record them -- it then reconstructs the nesting
 * from the timestamps. Writing them filled with a "missing" value instead
 * would be read as data: every call would share one id and the call graph
 * would collapse to a single node.
 *
 * A rank writes its own file, exactly as it writes its own .spt. Merging the
 * ranks of an MPI run stays a post-processing step
 * (`scope-profiler import-native rank*.h5 -o profile.h5`), so nothing here
 * needs MPI or a parallel HDF5 build.
 *
 * The file is published atomically: it is built at a sibling temporary path
 * and rename()d over the destination only once it has closed cleanly, so an
 * interrupted run leaves the previous profile intact rather than a truncated
 * one.
 */
#ifndef SCOPE_PROFILER_HDF5_H
#define SCOPE_PROFILER_HDF5_H

#include <hdf5.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* Mirrors h5writer._SUMMARY_DTYPE, field for field. h5py reads the compound
 * back as that numpy dtype, so member names, types and count must match. */
typedef struct {
    int64_t total;
    int64_t minimum;
    int64_t maximum;
    int64_t first;
    int64_t last;
    int64_t start_minimum;
    int64_t end_maximum;
    uint64_t gpu_count;
    int64_t gpu_total;
    double mean;
    double m2;
} sp_h5_summary;

/* h5writer._NO_EXCLUSIVE_TOTAL: an index row whose writer did not compute an
 * exclusive total. Unlike the event columns, this one is a per-row field that
 * has to be present, and the reader does check it for -1. */
#define SP_H5_NOT_RECORDED (-1)

/* Every handle a write needs, so one cleanup path can close whatever was
 * opened -- including on the first failure, halfway through. */
typedef struct {
    hid_t file;
    hid_t string_type;
    hid_t summary_type;
} sp_h5_context;

static hid_t sp_h5_string_type(void)
{
    hid_t type = H5Tcopy(H5T_C_S1);

    if (type < 0) {
        return -1;
    }
    if (H5Tset_size(type, H5T_VARIABLE) < 0 || H5Tset_cset(type, H5T_CSET_UTF8) < 0) {
        H5Tclose(type);
        return -1;
    }
    return type;
}

static hid_t sp_h5_summary_type(void)
{
    hid_t type = H5Tcreate(H5T_COMPOUND, sizeof(sp_h5_summary));
    int failed = 0;

    if (type < 0) {
        return -1;
    }
#define SP_H5_FIELD(name, member_type) \
    (failed |= H5Tinsert(type, #name, HOFFSET(sp_h5_summary, name), member_type) < 0)
    SP_H5_FIELD(total, H5T_NATIVE_INT64);
    SP_H5_FIELD(minimum, H5T_NATIVE_INT64);
    SP_H5_FIELD(maximum, H5T_NATIVE_INT64);
    SP_H5_FIELD(first, H5T_NATIVE_INT64);
    SP_H5_FIELD(last, H5T_NATIVE_INT64);
    SP_H5_FIELD(start_minimum, H5T_NATIVE_INT64);
    SP_H5_FIELD(end_maximum, H5T_NATIVE_INT64);
    SP_H5_FIELD(gpu_count, H5T_NATIVE_UINT64);
    SP_H5_FIELD(gpu_total, H5T_NATIVE_INT64);
    SP_H5_FIELD(mean, H5T_NATIVE_DOUBLE);
    SP_H5_FIELD(m2, H5T_NATIVE_DOUBLE);
#undef SP_H5_FIELD
    if (failed) {
        H5Tclose(type);
        return -1;
    }
    return type;
}

/* Attach a scalar attribute of `type` to `target`. */
static int sp_h5_scalar_attribute(hid_t target, const char *name, hid_t type, const void *value)
{
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attribute;
    int failed;

    if (space < 0) {
        return 1;
    }
    attribute = H5Acreate2(target, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(space);
    if (attribute < 0) {
        return 1;
    }
    failed = H5Awrite(attribute, type, value) < 0;
    H5Aclose(attribute);
    return failed;
}

static int sp_h5_int_attribute(hid_t target, const char *name, int64_t value)
{
    return sp_h5_scalar_attribute(target, name, H5T_NATIVE_INT64, &value);
}

static int sp_h5_string_attribute(hid_t target, hid_t string_type, const char *name, const char *value)
{
    /* A variable-length string attribute is written from a pointer *to* the
     * pointer: the element itself is the char*. */
    const char *element = value != NULL ? value : "";

    return sp_h5_scalar_attribute(target, name, string_type, &element);
}

/* Create a one-dimensional dataset of `count` elements and write `data` into
 * it. A NULL `data` creates the dataset without filling it, for a column
 * written afterwards one hyperslab at a time. */
static int sp_h5_write_dataset(
    hid_t parent,
    const char *name,
    hid_t type,
    hsize_t count,
    const void *data)
{
    hsize_t dims[1];
    hid_t space;
    hid_t dataset;
    int failed;

    dims[0] = count;
    space = H5Screate_simple(1, dims, NULL);
    if (space < 0) {
        return 1;
    }
    dataset = H5Dcreate2(parent, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset < 0) {
        H5Sclose(space);
        return 1;
    }
    failed = 0;
    if (count > 0 && data != NULL) {
        failed = H5Dwrite(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0;
    }
    H5Dclose(dataset);
    H5Sclose(space);
    return failed;
}

/* Create an empty int64 event column of `count` elements, to be filled one
 * region-sized hyperslab at a time by sp_h5_write_event_slice(). */
static int sp_h5_create_event_column(hid_t events, const char *name, hsize_t count)
{
    return sp_h5_write_dataset(events, name, H5T_NATIVE_INT64, count, NULL);
}

/* Write `count` int64 values at `offset` into an existing event column. */
static int sp_h5_write_event_slice(
    hid_t events,
    const char *name,
    hsize_t offset,
    hsize_t count,
    const int64_t *values)
{
    hsize_t start[1];
    hsize_t block[1];
    hid_t dataset;
    hid_t file_space;
    hid_t memory_space;
    int failed;

    if (count == 0) {
        return 0;
    }
    dataset = H5Dopen2(events, name, H5P_DEFAULT);
    if (dataset < 0) {
        return 1;
    }
    file_space = H5Dget_space(dataset);
    memory_space = H5Screate_simple(1, &count, NULL);
    start[0] = offset;
    block[0] = count;
    failed = file_space < 0 || memory_space < 0 ||
             H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, block, NULL) < 0 ||
             H5Dwrite(dataset, H5T_NATIVE_INT64, memory_space, file_space, H5P_DEFAULT, values) < 0;
    if (memory_space >= 0) {
        H5Sclose(memory_space);
    }
    if (file_space >= 0) {
        H5Sclose(file_space);
    }
    H5Dclose(dataset);
    return failed;
}

/* The fixed-size statistics for one region's events, matching
 * h5writer._timing_summary(). Two passes, because m2 is the sum of squared
 * deviations from the mean and the mean is not known until the first is
 * done; the alternative (a running Welford update) is what the reader uses to
 * *combine* these across ranks, not what the writer needs. */
static void sp_h5_summarize(
    const int64_t *starts,
    const int64_t *ends,
    int64_t count,
    sp_h5_summary *summary)
{
    int64_t i;
    double sum = 0.0;
    double m2 = 0.0;

    memset(summary, 0, sizeof(*summary));
    if (count <= 0) {
        return;
    }

    summary->minimum = ends[0] - starts[0];
    summary->maximum = summary->minimum;
    summary->first = summary->minimum;
    summary->last = ends[count - 1] - starts[count - 1];
    summary->start_minimum = starts[0];
    summary->end_maximum = ends[0];
    for (i = 0; i < count; ++i) {
        int64_t duration = ends[i] - starts[i];

        summary->total += duration;
        if (duration < summary->minimum) {
            summary->minimum = duration;
        }
        if (duration > summary->maximum) {
            summary->maximum = duration;
        }
        if (starts[i] < summary->start_minimum) {
            summary->start_minimum = starts[i];
        }
        if (ends[i] > summary->end_maximum) {
            summary->end_maximum = ends[i];
        }
        sum += (double)duration;
    }
    summary->mean = sum / (double)count;
    for (i = 0; i < count; ++i) {
        double deviation = (double)(ends[i] - starts[i]) - summary->mean;

        m2 += deviation * deviation;
    }
    summary->m2 = m2;
    /* No GPU timing in the C API: gpu_count/gpu_total stay 0, which is what
     * the reader reads as "this run measured no GPU work". */
}

/* An ISO-8601 UTC timestamp, the format metadata.collect_metadata() writes. */
static void sp_h5_timestamp(char *buffer, size_t size)
{
    time_t now = time(NULL);
    struct tm utc;

    if (gmtime_r(&now, &utc) == NULL || strftime(buffer, size, "%Y-%m-%dT%H:%M:%S+00:00", &utc) == 0) {
        snprintf(buffer, size, "unknown");
    }
}

/* The /metadata group. Only what a C process actually knows: the reader
 * treats a missing key as unmeasured, and merging ranks later fills in what
 * a Python side recorded. */
static int sp_h5_write_metadata(sp_h5_context *context, int rank, int64_t start_time_ns, const char *label)
{
    hid_t group = H5Gcreate2(context->file, "metadata", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    char hostname[256];
    char timestamp[64];
    int failed = 0;

    if (group < 0) {
        return 1;
    }
    if (gethostname(hostname, sizeof(hostname)) != 0) {
        snprintf(hostname, sizeof(hostname), "unknown");
    }
    hostname[sizeof(hostname) - 1] = '\0';
    sp_h5_timestamp(timestamp, sizeof(timestamp));

    /* "native" is what native_trace.load_traces() stamps on an imported run,
     * so a directly-written file and an imported one describe themselves the
     * same way. */
    failed |= sp_h5_string_attribute(group, context->string_type, "source", "native");
    failed |= sp_h5_string_attribute(group, context->string_type, "source_language", "c");
    failed |= sp_h5_int_attribute(group, "trace_format_version", SP_FORMAT_VERSION);
    failed |= sp_h5_int_attribute(group, "mpi_rank", rank);
    failed |= sp_h5_string_attribute(group, context->string_type, "hostname", hostname);
    failed |= sp_h5_string_attribute(group, context->string_type, "timestamp", timestamp);
    if (label != NULL) {
        failed |= sp_h5_string_attribute(group, context->string_type, "label", label);
    }
    if (start_time_ns >= 0) {
        /* The timeline origin, as a Python run records it at setup(). */
        failed |= sp_h5_int_attribute(group, "start_time_ns", start_time_ns);
    }
    H5Gclose(group);
    return failed;
}

/* Write one rank's schema-2 profile. `regions`/`counts` describe only the
 * regions with at least one completed call, in id order. */
static int sp_h5_write_profile(
    sp_h5_context *context,
    sp_profiler *profiler,
    sp_region_t **regions,
    const int64_t *counts,
    int num_regions,
    const char *label)
{
    hid_t index_group = -1;
    hid_t table_group = -1;
    hid_t events_group = -1;
    const char **strings = NULL;
    uint32_t *uint32_column = NULL;
    uint64_t *uint64_column = NULL;
    int64_t *int64_column = NULL;
    sp_h5_summary *summaries = NULL;
    hsize_t rows = (hsize_t)num_regions;
    hsize_t total_events = 0;
    int64_t start_time_ns = -1;
    int failed = 0;
    int i;

    for (i = 0; i < num_regions; ++i) {
        total_events += (hsize_t)counts[i];
        if (counts[i] > 0 && (start_time_ns < 0 || regions[i]->start_times[0] < start_time_ns)) {
            start_time_ns = regions[i]->start_times[0];
        }
    }

    /* One allocation per column shape, reused across the columns of that
     * shape: a row count and an event count, never a copy of the whole file. */
    strings = (const char **)malloc((rows > 0 ? (size_t)rows : 1) * sizeof(*strings));
    uint32_column = (uint32_t *)malloc((rows > 0 ? (size_t)rows : 1) * sizeof(*uint32_column));
    uint64_column = (uint64_t *)malloc((rows > 0 ? (size_t)rows : 1) * sizeof(*uint64_column));
    int64_column = (int64_t *)malloc((rows > 0 ? (size_t)rows : 1) * sizeof(*int64_column));
    summaries = (sp_h5_summary *)malloc((rows > 0 ? (size_t)rows : 1) * sizeof(*summaries));
    if (strings == NULL || uint32_column == NULL || uint64_column == NULL ||
        int64_column == NULL || summaries == NULL) {
        profiler->last_error = SP_ERR_NO_MEMORY;
        failed = 1;
        goto cleanup;
    }

    failed |= sp_h5_int_attribute(context->file, "scope_profiler_schema", 2);
    failed |= sp_h5_string_attribute(context->file, context->string_type, "storage_layout", "columnar");
    failed |= sp_h5_write_metadata(context, profiler->rank_id, start_time_ns, label);
    if (failed) {
        goto cleanup;
    }

    /* region_table: the name of every region id in this file. */
    table_group = H5Gcreate2(context->file, "region_table", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (table_group < 0) {
        failed = 1;
        goto cleanup;
    }
    for (i = 0; i < num_regions; ++i) {
        strings[i] = regions[i]->name;
    }
    failed |= sp_h5_write_dataset(table_group, "names", context->string_type, rows, strings);
    if (failed) {
        goto cleanup;
    }

    /* rank_region_index: one row per region, all of them this rank's. */
    index_group = H5Gcreate2(context->file, "rank_region_index", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (index_group < 0) {
        failed = 1;
        goto cleanup;
    }
    for (i = 0; i < num_regions; ++i) {
        uint32_column[i] = (uint32_t)i;
    }
    failed |= sp_h5_write_dataset(index_group, "region_ids", H5T_NATIVE_UINT32, rows, uint32_column);
    for (i = 0; i < num_regions; ++i) {
        uint32_column[i] = (uint32_t)profiler->rank_id;
    }
    failed |= sp_h5_write_dataset(index_group, "ranks", H5T_NATIVE_UINT32, rows, uint32_column);

    /* Regions are written back to back, so each row owns a consecutive slice
     * of the event columns. */
    {
        uint64_t offset = 0;

        for (i = 0; i < num_regions; ++i) {
            uint64_column[i] = offset;
            offset += (uint64_t)counts[i];
        }
    }
    failed |= sp_h5_write_dataset(index_group, "event_offsets", H5T_NATIVE_UINT64, rows, uint64_column);
    for (i = 0; i < num_regions; ++i) {
        uint64_column[i] = (uint64_t)counts[i];
    }
    failed |= sp_h5_write_dataset(index_group, "event_counts", H5T_NATIVE_UINT64, rows, uint64_column);

    for (i = 0; i < num_regions; ++i) {
        int64_column[i] = regions[i]->source_file != NULL ? regions[i]->source_line : -1;
    }
    failed |= sp_h5_write_dataset(index_group, "source_lines", H5T_NATIVE_INT64, rows, int64_column);
    for (i = 0; i < num_regions; ++i) {
        /* Nesting is not recorded here, so exclusive time is left for the
         * reader to reconstruct from the timestamps, exactly as it does for
         * an imported trace. */
        int64_column[i] = SP_H5_NOT_RECORDED;
    }
    failed |= sp_h5_write_dataset(index_group, "exclusive_totals", H5T_NATIVE_INT64, rows, int64_column);

    for (i = 0; i < num_regions; ++i) {
        strings[i] = regions[i]->source_file != NULL ? regions[i]->source_file : "";
    }
    failed |= sp_h5_write_dataset(index_group, "source_files", context->string_type, rows, strings);
    for (i = 0; i < num_regions; ++i) {
        strings[i] = "";
    }
    failed |= sp_h5_write_dataset(index_group, "source_texts", context->string_type, rows, strings);
    for (i = 0; i < num_regions; ++i) {
        strings[i] = "[]";
    }
    failed |= sp_h5_write_dataset(index_group, "tags", context->string_type, rows, strings);

    for (i = 0; i < num_regions; ++i) {
        sp_h5_summarize(regions[i]->start_times, regions[i]->end_times, counts[i], &summaries[i]);
    }
    failed |= sp_h5_write_dataset(
        index_group, "summary_statistics", context->summary_type, rows, summaries);
    if (failed) {
        goto cleanup;
    }

    /* events: every region's timestamps, concatenated in row order. */
    events_group = H5Gcreate2(context->file, "events", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (events_group < 0) {
        failed = 1;
        goto cleanup;
    }
    failed |= sp_h5_create_event_column(events_group, "start_times", total_events);
    failed |= sp_h5_create_event_column(events_group, "end_times", total_events);
    if (failed) {
        goto cleanup;
    }

    /* Only the two columns this API records. Everything else the schema
     * allows per call -- call/parent ids, GPU durations, thread and task
     * lanes -- is left out, which is what tells the reader to derive the
     * nesting from these timestamps instead of trusting ids that were never
     * assigned. */
    {
        hsize_t offset = 0;

        for (i = 0; i < num_regions && !failed; ++i) {
            hsize_t count = (hsize_t)counts[i];

            failed |= sp_h5_write_event_slice(
                events_group, "start_times", offset, count, regions[i]->start_times);
            failed |= sp_h5_write_event_slice(
                events_group, "end_times", offset, count, regions[i]->end_times);
            offset += count;
        }
    }

cleanup:
    if (events_group >= 0) {
        H5Gclose(events_group);
    }
    if (index_group >= 0) {
        H5Gclose(index_group);
    }
    if (table_group >= 0) {
        H5Gclose(table_group);
    }
    free(strings);
    free(uint32_column);
    free(uint64_column);
    free(int64_column);
    free(summaries);
    return failed;
}

/* The label a run gets in summaries and plots: the prefix's last path
 * component, so "out/run3" is reported as "run3". Returns a pointer into
 * `prefix`, or NULL if there is nothing usable. */
static const char *sp_h5_label(const char *prefix)
{
    const char *separator;

    if (prefix == NULL || *prefix == '\0') {
        return NULL;
    }
    separator = strrchr(prefix, '/');
    return separator != NULL ? (separator[1] != '\0' ? separator + 1 : NULL) : prefix;
}

/* Write "<prefix>_rank<NNNNN>.h5", the counterpart of write_trace(). Does not
 * modify `profiler`, beyond its cached path and last_error. */
static int write_hdf5(sp_profiler *profiler)
{
    sp_h5_context context;
    sp_region_t **regions = NULL;
    int64_t *counts = NULL;
    const char *path;
    char *temporary = NULL;
    size_t temporary_size;
    int num_regions = 0;
    int failed = 0;
    int i;

    context.file = -1;
    context.string_type = -1;
    context.summary_type = -1;

    path = output_path(profiler);
    if (path == NULL) {
        fprintf(stderr, "scope_profiler: out of memory writing the profile\n");
        profiler->last_error = SP_ERR_NO_MEMORY;
        return 1;
    }

    if (profiler->n_regions > 0) {
        regions = (sp_region_t **)malloc((size_t)profiler->n_regions * sizeof(*regions));
        counts = (int64_t *)malloc((size_t)profiler->n_regions * sizeof(*counts));
        if (regions == NULL || counts == NULL) {
            fprintf(stderr, "scope_profiler: out of memory writing the profile\n");
            profiler->last_error = SP_ERR_NO_MEMORY;
            free(regions);
            free(counts);
            return 1;
        }
    }
    for (i = 0; i < profiler->n_regions; ++i) {
        int64_t count = written_count(&profiler->regions[i]);

        if (count > 0) {
            regions[num_regions] = &profiler->regions[i];
            counts[num_regions] = count;
            num_regions += 1;
        }
    }

    /* Build beside the destination and rename over it, so a run killed
     * mid-write leaves the previous profile readable instead of a truncated
     * file. */
    temporary_size = strlen(path) + 32;
    temporary = (char *)malloc(temporary_size);
    if (temporary == NULL) {
        fprintf(stderr, "scope_profiler: out of memory writing %s\n", path);
        profiler->last_error = SP_ERR_NO_MEMORY;
        free(regions);
        free(counts);
        return 1;
    }
    snprintf(temporary, temporary_size, "%s.tmp%d", path, (int)getpid());

    context.string_type = sp_h5_string_type();
    context.summary_type = sp_h5_summary_type();
    context.file = context.string_type < 0 || context.summary_type < 0
                       ? -1
                       : H5Fcreate(temporary, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (context.file < 0) {
        fprintf(stderr, "scope_profiler: cannot write %s\n", path);
        profiler->last_error = SP_ERR_IO;
        failed = 1;
    } else {
        failed = sp_h5_write_profile(
            &context, profiler, regions, counts, num_regions, sp_h5_label(profiler->output_prefix));
        if (H5Fclose(context.file) < 0) {
            failed = 1;
        }
        context.file = -1;
    }

    if (!failed && rename(temporary, path) != 0) {
        fprintf(stderr, "scope_profiler: cannot publish %s\n", path);
        failed = 1;
    }
    if (failed) {
        remove(temporary);
        if (profiler->last_error != SP_ERR_NO_MEMORY) {
            profiler->last_error = SP_ERR_IO;
        }
        fprintf(stderr, "scope_profiler: failed to write %s\n", path);
    }

    if (context.summary_type >= 0) {
        H5Tclose(context.summary_type);
    }
    if (context.string_type >= 0) {
        H5Tclose(context.string_type);
    }
    free(temporary);
    free(regions);
    free(counts);
    return failed;
}

#endif /* SCOPE_PROFILER_HDF5_H */
