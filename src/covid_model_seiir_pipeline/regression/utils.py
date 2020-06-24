from slime.core.data import MRData
import numpy as np


COL_T = 'days'
COL_BETA = 'beta'
COL_GROUP = 'loc_id'


def convert_inputs_for_beta_model(data_cov, df_beta, covmodel_set):
    df_cov, col_t_cov, col_group_cov = data_cov
    df = df_beta.merge(
        df_cov, 
        left_on=[COL_T, COL_GROUP], 
        right_on=[col_t_cov, col_group_cov],
    ).copy()
    df.sort_values(inplace=True, by=[COL_GROUP, COL_T])
    cov_names = [covmodel.col_cov for covmodel in covmodel_set.cov_models]
    mrdata = MRData(df, col_group=COL_GROUP, col_obs=COL_BETA, col_covs=cov_names)

    return mrdata


def convolve_mean(mat, radius=None):
    """Convolve mean a 2D matrix by given radius.
    Args:
        mat (numpy.ndarray):
            Matrix of interest.
        radius (arraylike{int} | None, optional):
            Given radius, if None assume radius = (0, 0).
    Returns:
        numpy.ndarray:
            The convolved sum, with the same shape with original matrix.
    """
    mat = np.array(mat).astype(float)
    assert mat.ndim == 2
    if radius is None:
        return mat
    assert hasattr(radius, '__iter__')
    radius = np.array(radius).astype(int)
    assert radius.size == 2
    assert all([r >= 0 for r in radius])
    # import pdb; pdb.set_trace()
    shape = np.array(mat.shape)
    window_shape = tuple(radius*2 + 1)

    mat = np.pad(mat, ((radius[0],),
                       (radius[1],)), 'constant', constant_values=np.nan)
    view_shape = tuple(np.subtract(mat.shape, window_shape) + 1) + window_shape
    strides = mat.strides*2
    sub_mat = np.lib.stride_tricks.as_strided(mat, view_shape, strides)
    sub_mat = sub_mat.reshape(*shape, np.prod(window_shape))

    return np.nanmean(sub_mat, axis=2)
