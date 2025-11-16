#!/usr/bin/env python

import pylikwid

pylikwid.markerinit()
pylikwid.markerthreadinit()
liste = []
pylikwid.markerstartregion("listappend")
for i in range(0, 1000000):
    liste.append(i)
pylikwid.markerstopregion("listappend")
nr_events, eventlist, time, count = pylikwid.markergetregion("listappend")
for i, e in enumerate(eventlist):
    print(i, e)
pylikwid.markerclose()
