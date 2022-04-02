import functools
from typing import Any, Callable, List, Optional

import pandas as pd
from pathos import multiprocessing
import tqdm


Loader = Callable[[Any, Optional[pd.Index], int, int, bool], pd.DataFrame]


def run_parallel(runner: Callable,
                 arg_list: List,
                 num_cores: int,
                 progress_bar: bool = False) -> List[Any]:
    """Runs a single argument function in parallel over a list of arguments.

    This function dodges multiprocessing if only a single process is requested to
    make functions more flexible to debugging. It also supports progress bars if
    requested.

    """
    if num_cores == 1:
        result = []
        for arg in tqdm.tqdm(arg_list, disable=not progress_bar):
            result.append(runner(arg))
    else:
        with multiprocessing.ProcessPool(num_cores) as pool:
            result = list(tqdm.tqdm(
                pool.imap(runner, arg_list),
                total=len(arg_list),
                disable=not progress_bar,
            ))
    return result


def make_loader(loader: Callable[[int, Optional[str]], pd.DataFrame],
                measure: str,
                measure_version: str = None):
    """Makes a loader from a draw-specific data_interface method that returns all draws as a dataframe.

    Parameters
    ----------
    loader
        The relevant loading method on the data interface CLASS. When making loaders, we don't
        have access to the data interface instance yet, but the method off the class will be used
        to find the appropriate instance method when available.
    measure
        An individual column in the dataset produced by the provided loader
    measure_version
        Which version of the past (case, death, admission, final) to load.

    Returns
    -------
    Loader
        A caller that takes a data interface and some parameters and produces
        a dataframe with all requested draws.

    """

    def _loader(data_interface,
                *,  # Disallow other positional args.
                index: pd.Index = None,
                num_draws: int = 1,
                num_cores: int = 1,
                progress_bar: bool = False) -> pd.DataFrame:
        """Loads draws of a measure stored in separate files and concats them into a single dataset.

        Parameters
        ----------
        data_interface

            Function that takes as an argument a draw id and an optional list of columns
            and loads those columns out of a draw specific dataset.
        index
            An optional index used to standardize the loaded dataset when it is loaded.
            Using a shared index dramatically speeds up concatenation of the draw level
            datasets.
        num_draws
            The number of draws to load. No bounds checking is done to ensure all draws
            requested are available to load.
        num_cores
            The number of processes to use to load the datasets. If >1, multiprocessing
            will be used to load the data.
        progress_bar
            Whether to display a progress bar when loading the data.

        Returns
        -------
        pd.DataFrame
            A single dataset with a column for each requested draw.

        """
        _runner = functools.partial(
            draw_runner,
            loader=getattr(data_interface, loader.__name__),
            index=index,
            measure_version=measure_version,
            column=measure,
        )
        result = run_parallel(
            runner=_runner,
            arg_list=list(range(num_draws)),
            num_cores=num_cores,
            progress_bar=progress_bar,
        )
        result = pd.concat(result, axis=1)
        result.columns = [f'draw_{draw}' for draw in range(num_draws)]
        return result

    return _loader


def draw_runner(draw_id: int,
                loader: Callable[[int, Optional[List[str]]], pd.DataFrame],
                measure_version: Optional[str],
                column: Optional[str],
                index: Optional[pd.Index]) -> pd.Series:
    """Loads a column from a draw level dataset and re-indexes the dataset if appropriate.

    Parameters
    ----------
    draw_id
        The draw to load.
    loader
        Function that takes as an argument a draw id and an optional list of columns
        and loads those columns out of a draw specific dataset.
    measure_version
        Which version of the past (case, death, admission, final) to load.
    column
        An optional column to load from the dataset produced by the loader. The returned
        dataset must be a single column, so this argument can be used to load from a
        multi-column draw level data set.
    index
        An optional index used to standardize the loaded dataset when it is loaded.
        Using a shared index dramatically speeds up concatenation of the draw level
        datasets.

    """
    if column and 'round' in index.names:
        columns = [column, 'round']
    elif column:
        columns = [column]
    else:
        columns = None

    if measure_version is not None:
        data = loader(draw_id, measure_version=measure_version, columns=columns)
    else:
        data = loader(draw_id, columns=columns)
    if index is not None:
        data = data.reset_index().set_index(index.names).reindex(index)
    return data.squeeze()
