import type {
  BuildOptions,
  Figure,
  PlotlyLike,
  ThemeName,
  ThemeTokens,
} from "./index.js";

export function parseRegionFilter(text: string | null | undefined): string[];
export function matchesRegionFilter(
  region: string | null | undefined,
  terms: string[],
): boolean;
export function regionFilter(
  text: string | null | undefined,
): ((region: string) => boolean) | undefined;
export function documentTheme(): ThemeName;

export interface FigureRegistryOptions {
  /** A theme, or a function returning one, read at each render. */
  theme?: ThemeName | ThemeTokens | (() => ThemeName | ThemeTokens);
  /** Plotly config merged into every render. */
  config?: object;
  /** Figure builder, defaulting to `buildFigure`. */
  build?: (payload: object, options?: BuildOptions) => Figure;
}

export interface RenderOptions extends BuildOptions {
  /** A filter box's contents, turned into `filterRegion` for this figure. */
  regionFilter?: string;
  /** Builder for this figure only, overriding the registry's. */
  build?: (payload: object, options?: BuildOptions) => Figure;
}

export interface FigureRegistry {
  render(
    container: Element | string | null | undefined,
    payload: object,
    options?: RenderOptions,
  ): Promise<unknown> | undefined;
  refresh(): Promise<unknown[]>;
  forget(container: Element | string): boolean;
  clear(): void;
  readonly size: number;
  watch(target?: EventTarget, event?: string): () => void;
}

export function createFigureRegistry(
  plotly: PlotlyLike,
  options?: FigureRegistryOptions,
): FigureRegistry;
