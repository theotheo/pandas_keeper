import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from pandas.io.json import json_normalize



def var_to_name(var, items):
    return [k for k, v in locals().items() if v is var][0]


def fn_to_df(fn, params={}):
    ext = fn.split('.')[-1]
    format_to_method = {
        'pkl': ('pickle', {}),
        'csv': ('csv', {}),
        'json': ('json', {}),
        'jsonlines': ('json', {'lines': True}),
        'jsonl': ('json', {'lines': True})
    }
    method, params_ = format_to_method[ext]
    params.update(params_)
    read = getattr(pd, 'read_{}'.format(method))
    return read(fn, **params)


def col_metatypes(df):
    res = {}
    for col in df.columns:
        if np.any(df[col].apply(lambda x: type(x) is dict)):
            kind = 'dict'
        elif np.any(df[col].apply(lambda x: type(x) is list)):
            kind = 'list'
        elif str(df[col].dtype) == 'category':
            kind = 'category'
        elif np.any(df[col].apply(lambda x: type(x) is str)):
            kind = 'str'
        # elif np.any(df[col].apply(lambda x: type(x) is str)):
        #     kind = str
        else:
            kind = df[col].dtype
        res[col] = kind
    return res


def fvdf(df, n=2):
    def colorize_dtype(x):
        dtype_2_color = {
            'object': '#bbbb00',
            'dict': '#660000',
            'list': '#330000',
            'bool': '#222222',
            'float64': '#333333',
            'float32': '#444444',
            'int64': '#555555',
            'int32': '#666666',
            'datetime64[ns]': '#003000',
            'str': '#0000aa',
            'category': '#007000'
        }
        color = dtype_2_color.get(str(x), 'None')

        return 'background-color: {}'.format(color)

    nrow, ncol = df.shape
    metatypes = col_metatypes(df)

    subset = df[[col for col, typ in metatypes.items() if str(typ) not in ['list', 'dict']]]
    describe_df = subset.describe(percentiles=[], include='all').T
    cols = list(set(describe_df.columns).intersection(['count', 'unique', 'mean']))
    describe_df = describe_df[cols]
    if 'mean' not in cols:
        describe_df['mean'] = np.nan
    renamed_cols = ['_' + col for col in describe_df.columns]
    describe_df.columns = renamed_cols
    describe_df = describe_df.fillna({'_unique': '', '_mean': ''})

    dtypes_df = pd.DataFrame.from_dict(metatypes, orient='index')
    dtypes_df.columns = ['_dtype']
    print(dtypes_df)
    head = pd.concat([dtypes_df, describe_df, df.head(n).T, df[nrow//2:nrow//2+n].T, df.tail(n).T], axis=1)

    # http://pandas-docs.github.io/pandas-docs-travis/style.html
    # TODO: doesn't work
    def hover(hover_color="#ffff99"):
        return dict(selector="tr:hover",
            props=[("background-color", "%s" % hover_color)])
    styles = [
        hover()
    ]
    try:

        head = (head.style.applymap(colorize_dtype, subset=['_dtype'])
                    .bar(subset=['_count', '_unique'])
                    .set_properties(**{
                           'color': 'black',
                           'border-color': 'white'})
                    .set_table_styles(styles))
    except ValueError as e:
        print('Error: {}'.format(e))
# if ncol < 10:
#     head = head.T

    return head


def split_to_dfs(df, id_key='id', root='root'):
    root_id = '_'.join([root, id_key])
    logger.info('root id is "{}"\n'.format(root_id))

    def rename_index(df):
        df = df.rename(columns={id_key: root_id})
        return df.set_index(root_id)

    df = df.rename(columns={id_key: root_id})
    logger.debug(list(df.columns))
    kinds = col_metatypes(df)
    logger.debug(kinds)

    dfs = {}

    for col, kind in kinds.items():
        kind = str(kind)
        if kind == 'dict' or kind == 'list':
            # fillna http://stackoverflow.com/questions/25898888/pandas-replace-all-nan-values-in-dataframe-with-empty-python-dict-objects
            if kind == 'dict':
                df[col] = df[col].apply(lambda x: {} if (type(x) is not list) and (pd.isnull(x)) else x)
            elif kind == 'list':
                df[col] = df[col].apply(lambda x: [] if (type(x) is not list and pd.isnull(x)) else x)
            data = df.reset_index().ix[:, [col, root_id]].to_dict(orient='row')
            # NB: sep='__' for bcolz
            if kind == 'dict':
                extracted_df = json_normalize(data, meta=[root_id], sep='__')
            elif kind == 'list':
                extracted_df = json_normalize(data, col, meta=[root_id], sep='__').rename(columns={0: 'item'})
            # extracted_df = rename_index(extracted_df)

            print("{} '{}' => ({}) {}".format(kind, col, extracted_df.shape[0], list(extracted_df.columns),))
            dfs[col] = extracted_df

            # replace current column in df with True/None for dict object and length for list
#             if kind == dict:
#                 df[col] = np.where(df[col] == {}, None, True) # dict
#             elif kind == list:
#                 df[col] = df[col].apply(len) # list
            df = df.drop(col, axis=1)

    df = df.set_index(root_id)
    print("'{}' => {}".format(root, list(df.columns)))
    dfs[root] = df

    return dfs


def join_dfs(dfs, root_df_key='root'):
    root = dfs[root_df_key]
    root_id = root_df_key + '_id'
    for k, d in dfs.items():
        if k != root_df_key:
            colnames = {col: '{}_{}'.format(k, col) for col in d.columns if col != root_id}
            print(colnames)
            d = d.rename(columns=colnames)
            print(d.columns)
            root = d.join(root, on=root_id).set_index(root_id)
            logger.debug(root.columns)
    # root = root.reset_index()
    root.set_index(root_id)
    return root


def pivoted(df, col, root_id='root_id'):
    col = 'visited'
    data = df.ix[:, [col, root_id]].to_dict(orient='row')

    extracted_df = json_normalize(data, col, meta=[root_id])
    pivoted = pd.pivot_table(extracted_df, index=root_id, columns=[0],  aggfunc='size', fill_value=0)
    return pivoted


def col_metatypes(df):
    res = {}
    for col in df.columns:
        if np.any(df[col].apply(lambda x: type(x) is dict)):
            kind = 'dict'
        elif np.any(df[col].apply(lambda x: type(x) is list)):
            kind = 'list'
        elif str(df[col].dtype) == 'category':
            kind = 'category'
        elif np.any(df[col].apply(lambda x: type(x) is str)):
            kind = 'str'
        # elif np.any(df[col].apply(lambda x: type(x) is str)):
        #     kind = str
        else:
            kind = df[col].dtype
        res[col] = kind
    return res


def fvdf(df, n=2):
    def colorize_dtype(x):
        dtype_2_color = {
            'object': '#bbbb00',
            'dict': '#660000',
            'list': '#330000',
            'bool': '#222222',
            'float64': '#333333',
            'float32': '#444444',
            'int64': '#555555',
            'int32': '#666666',
            'datetime64[ns]': '#003000',
            'str': '#0000aa',
            'category': '#007000'
        }
        color = dtype_2_color.get(str(x), 'None')

        return 'background-color: {}'.format(color)

    nrow, ncol = df.shape
    metatypes = col_metatypes(df)

    subset = df[[col for col, typ in metatypes.items() if str(typ) not in ['list', 'dict']]]
    describe_df = subset.describe(percentiles=[], include='all').T
    cols = list(set(describe_df.columns).intersection(['count', 'unique', 'mean']))
    describe_df = describe_df[cols]
    if 'mean' not in cols:
        describe_df['mean'] = np.nan
    renamed_cols = ['_' + col for col in describe_df.columns]
    describe_df.columns = renamed_cols
    describe_df = describe_df.fillna({'_unique': 0, '_mean': '', '_count': 0})

    dtypes_df = pd.DataFrame.from_dict(metatypes, orient='index')
    dtypes_df.columns = ['_dtype']
    # print(describe_df)
    head = pd.concat([dtypes_df, describe_df, df.head(n).T, df[nrow//2:nrow//2+n].T, df.tail(n).T], axis=1)

    # http://pandas-docs.github.io/pandas-docs-travis/style.html
    # TODO: doesn't work
    def hover(hover_color="#ffff99"):
        return dict(selector="tr:hover",
            props=[("background-color", "%s" % hover_color)])
    styles = [
        hover()
    ]
    try:

        head = (head.style.applymap(colorize_dtype, subset=['_dtype'])
                    .bar(subset=['_count', '_unique'])
                    .set_properties(**{
                           'color': 'black',
                           'border-color': 'white'})
                    .set_table_styles(styles))
    except ValueError as e:
        print('Error: {}'.format(e))
# if ncol < 10:
#     head = head.T

    return head


def split_to_dfs(df, id_key='id', root='root'):
    root_id = '_'.join([root, id_key])
    print('root id is "{}"\n'.format(root_id))

    def rename_index(df):
        df = df.rename(columns={id_key: root_id})
        return df.set_index(root_id)

    df = df.rename(columns={id_key: root_id})
#     print(list(df.columns))
    kinds = col_metatypes(df)
    # print(kinds)

    dfs = {}

    for col, kind in kinds.items():
        kind = str(kind)
        if kind == 'dict' or kind == 'list':
            # fillna http://stackoverflow.com/questions/25898888/pandas-replace-all-nan-values-in-dataframe-with-empty-python-dict-objects
            if kind == 'dict':
                df[col] = df[col].apply(lambda x: {} if (type(x) is not list) and (pd.isnull(x)) else x)
            elif kind == 'list':
                df[col] = df[col].apply(lambda x: [] if (type(x) is not list and pd.isnull(x)) else x)
            data = df.reset_index().ix[:, [col, root_id]].to_dict(orient='row')
            # print(data)
            # NB: sep='__' for bcolz
            if kind == 'dict':
                extracted_df = json_normalize(data, meta=[root_id], sep='__')
            elif kind == 'list':
                extracted_df = json_normalize(data, col, meta=[root_id], sep='__').rename(columns={0: 'item'})
            # extracted_df = rename_index(extracted_df)

            print("{} '{}' => ({}) {}".format(kind, col, extracted_df.shape[0], list(extracted_df.columns),))
            dfs[col] = extracted_df

            # replace current column in df with True/None for dict object and length for list
#             if kind == dict:
#                 df[col] = np.where(df[col] == {}, None, True) # dict
#             elif kind == list:
#                 df[col] = df[col].apply(len) # list
            df = df.drop(col, axis=1)

    df = df.set_index(root_id)
    print("'{}' => {}".format(root, list(df.columns)))
    dfs[root] = df

    return dfs

def list_to_df(series):
    return series.apply(pd.Series).stack().reset_index(level=1, drop=True).apply(pd.Series)

def dict_to_df(series):
    return series.apply(pd.Series)

def split_to_dfs(df, id_key='id', root='root'):
    kinds = col_metatypes(df)
    # print(kinds)

    dfs = {}
    
    for col, kind in kinds.items():
        kind = str(kind)
        if kind == 'dict' or kind == 'list':
            # fillna http://stackoverflow.com/questions/25898888/pandas-replace-all-nan-values-in-dataframe-with-empty-python-dict-objects
            if kind == 'dict':
                df[col] = df[col].apply(lambda x: {} if (type(x) is not list) and (pd.isnull(x)) else x)
            elif kind == 'list':
                df[col] = df[col].apply(lambda x: [] if (type(x) is not list and pd.isnull(x)) else x)
            # data = df.reset_index().ix[:, [col, root_id]].to_dict(orient='row')
            # print(data)
            # NB: sep='__' for bcolz
            if kind == 'list':
                extracted_df = list_to_df(df[col])
            elif kind == 'dict':
                extracted_df = dict_to_df(df[col])
            # extracted_df = rename_index(extracted_df)

            print("{} '{}' => ({}) {}".format(kind, col, extracted_df.shape[0], list(extracted_df.columns),))
            dfs[col] = extracted_df

            # replace current column in df with True/None for dict object and length for list
#             if kind == dict:
#                 df[col] = np.where(df[col] == {}, None, True) # dict
#             elif kind == list:
#                 df[col] = df[col].apply(len) # list
            df = df.drop(col, axis=1)

    # df = df.set_index(root_id)
    print("'{}' => {}".format(root, list(df.columns)))
    dfs[root] = df

    return dfs


def join_dfs(dfs, root_df_key='root'):
    root = dfs[root_df_key]
    root_id = root_df_key + '_id'
    for k, d in dfs.items():
        if k != root_df_key:
            colnames = {col: '{}_{}'.format(k, col) for col in d.columns if col != root_id}
            print(colnames)
            d = d.rename(columns=colnames)
            print(d.columns)
            root = d.join(root, on=root_id).set_index(root_id)
            # print(root.columns)
    # root = root.reset_index()
    root.set_index(root_id)
    return root


def pivoted(df, col, root_id='root_id'):
    col = 'visited'
    data = df.ix[:, [col, root_id]].to_dict(orient='row')
    # print(data)
    extracted_df = json_normalize(data, col, meta=[root_id])
    pivoted = pd.pivot_table(extracted_df, index=root_id, columns=[0],  aggfunc='size', fill_value=0)
    return pivoted
