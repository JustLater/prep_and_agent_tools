from pydantic import BaseModel, Field
from typing import Optional, Tuple, List, Dict, Any, Union
import re
import base64
from pathlib import Path
import pandas as pd
import numpy as np
import os
import traceback
import json
import math
import string
from collections import Counter, OrderedDict

from smolagents import CodeAgent, FinalAnswerTool, load_tool, tool, Tool
from openai import OpenAI, DefaultHttpxClient

import datetime
import requests
import pytz
import yaml
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

import prepfile
from prepfile import DataStorage


def write_text_to_file(text: str, path_to_file: str):
    """Запись результата инструмента в файл"""
    try:
        with open(path_to_file + '/tmp.txt', 'r', encoding='utf-8') as file:
            existing_content = file.read()
    except FileNotFoundError:
        with open(path_to_file + '/tmp.txt', 'w', encoding='utf-8') as file:
            file.write(text)
        return True
    else:
        if existing_content == text:
            return False
        else:   
            with open(path_to_file + '/tmp.txt', 'a', encoding='utf-8') as file:
                file.write(text + '\n')
            return True

def generate_critics(model: str, base_url: str, api_key: str, report: str, existing_content: str, max_tokens=10000):
    """Генерирует ответ критика"""

    critics_promt = f"""prompt"""


    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    dict_1 = {
        "role": "user",
        "content": critics_promt + report,
    }

    completion = client.chat.completions.create(
        model=model, messages=[dict_1]
    )

    return(completion.choices[0].message.content)

def find_correlations(data_storage: DataStorage, threshold: float = 0.38) -> Dict[str, Dict[str, float]]:
    """
    Computes correlations between test columns.

    Args:
        data_storage: DataStorage class object containing prepared data
            (must contain encoded test dataframe and target dataframe)
        threshold: Correlation threshold for filtering results. Default is 0.4.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary with correlations in the format:
            {target_col: {question: corr_value}}

    """

    if not hasattr(data_storage, 'test_ohe_df') or not hasattr(data_storage, 'target_df'):
        raise ValueError("DataStorage object must have 'test_ohe_df' and 'target_df' attributes")

    input_df = data_storage.test_ohe_df
    target_df = data_storage.target_df

    if input_df.empty or target_df.empty:
        raise ValueError("Input or target DataFrame is empty")

    correlations = {}
    questions = set(col.split('_')[0] for col in input_df.columns if col not in target_df.columns)

    for target_col in target_df.columns:
        target_corrs = {}

        for question in questions:
            question_columns = [col for col in input_df.columns if col.startswith(question + '_')]

            if not question_columns:
                continue

            best_corr = 0

            for col in question_columns:
                corr = input_df[col].corr(target_df[target_col])
                if pd.isna(corr):
                    continue
                if abs(corr) >= abs(best_corr):
                    best_corr = corr

            if abs(best_corr) >= threshold:
                target_corrs[question] = best_corr

        sorted_correlations = sorted(
            target_corrs.items(),
            key=lambda x: int(x[0].split('.')[0]) if '.' in x[0] and x[0].split('.')[0].isdigit() else 0)

        correlations[target_col] = dict(sorted_correlations)

    return correlations

def count_answers_corr(data_storage: DataStorage, threshold: float = 0.38) -> pd.Series:
    """
    Computes correlations between the number of answers in questions and target variables.

    Automatically identifies columns with multiple answers in a single cell (separated by '\\n'),
    counts the number of answers for each question, and calculates Pearson correlation
    with each target variable.

    Args:
        data_storage: DataStorage class object containing prepared data
            (must contain test dataframe and target dataframe)
        threshold: Correlation threshold for filtering results. Default is 0.4.

    Returns:
        Series with correlations in the format {"question_name... vs target_name...": correlation_value}.
        Correlations are sorted by absolute value (in descending order).
    """

    if not hasattr(data_storage, 'test_df') or not hasattr(data_storage, 'target_df'):
        raise ValueError("DataStorage object must have 'test_df' and 'target_df' attributes")

    df = data_storage.test_df
    tar_df = data_storage.target_df

    multi_cols = []
    for col in df.columns:
        if col not in tar_df.columns:
            sample = df[col].astype(str).dropna().head(100)
            if any(sample.str.contains('\n', na=False)):
                multi_cols.append(col)

    if not multi_cols:
        print("Не найдено колонок с несколькими ответами в одной ячейке")
        return pd.Series()

    count_df = df[multi_cols + list(tar_df.columns)].copy()

    for col in multi_cols:
        count_df[col] = df[col].astype(str).apply(lambda x: len(str(x).split('\n')) if pd.notna(x) else 0)

    count_df = count_df.dropna(subset=tar_df.columns)

    correlations = pd.Series(dtype=float)

    for target_col in tar_df.columns:
        target = count_df[target_col]
        for col in multi_cols:
            try:
                corr, _ = pearsonr(count_df[col], target)
                if abs(corr) >= threshold:
                    correlations[f"{col[:30]}... vs {target_col[:30]}..."] = corr
            except Exception as e:
                print(f"Ошибка при расчете корреляции для столбца '{col}': {str(e)}")
                correlations[f"{col[:30]}... vs {target_col[:30]}..."] = np.nan

    return correlations

def all_show_heatmap(data_storage: DataStorage, threshold: float = 0.38) -> Dict[Tuple[str, str], float]:
    """
    Computes correlations between questions and target variables for a heatmap.

    Returns only significant correlations between different questions.

    Args:
        data_storage: DataStorage class object containing prepared data
            (must contain encoded test dataframe and target dataframe)
        threshold: Correlation threshold for filtering results. Default is 0.4.

    Returns:
        Dict[Tuple[str, str], float]: Dictionary in the format {('question1', 'question2'): correlation_value},
        containing only significant correlations between different questions.
    """

    if not hasattr(data_storage, 'test_ohe_df') or not hasattr(data_storage, 'target_df'):
        raise ValueError("DataStorage object must have 'test_ohe_df' and 'target_df' attributes")

    input_df = data_storage.test_ohe_df
    target_df = data_storage.target_df

    question_columns = [col.split('_')[0] for col in input_df.columns if '_' in col]
    target_columns = target_df.columns

    filtered_cols = []
    for col in input_df.columns:
        base_col = col.split('_')[0]
        if col in target_columns or (base_col not in target_columns and '_' in col):
            filtered_cols.append(col)

    corr_matrix = input_df[filtered_cols].corr()

    significant_corrs = {}
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:
                corr_value = corr_matrix.iloc[i, j]
                base_col1 = col1.split('_')[0]
                base_col2 = col2.split('_')[0]
                if abs(corr_value) >= threshold and base_col1 != base_col2:
                    significant_corrs[(col1, col2)] = corr_value

    return significant_corrs

def find_nlp_corrs(data_storage: DataStorage, threshold: float = 0.2) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Function for analyzing correlation between words from encoded dictionary and target variables.

    Args:
        data_storage: DataStorage class object containing prepared data
            (must contain encoded NLP dictionary and target dataframe)
        threshold: Correlation threshold for filtering results. Default is 0.2.

    Returns:
        Dict[str, Dict[str, Dict[str, float]]]: Dictionary with correlation results for each target column.
    """

    if not hasattr(data_storage, 'nlp_ohe_dict') or not hasattr(data_storage, 'target_df'):
        raise ValueError("DataStorage object must have 'nlp_ohe_dict' and 'target_df' attributes")

    ohe_dict = data_storage.nlp_ohe_dict
    target_df = data_storage.target_df

    corr_results = {}

    for target_col in target_df.columns:
        target_series = target_df[target_col]
        target_results = {}

        for df_name, ohe_df in ohe_dict.items():
            df_results = {}

            for word in ohe_df.columns:
                feature = ohe_df[word].dropna()
                common_index = target_series.index.intersection(feature.index)

                if len(common_index) > 1:
                    target_aligned = target_series.loc[common_index]
                    feature_aligned = feature.loc[common_index]

                    # print(f"Сюда забежали! Отладка: {target_aligned} и {feature_aligned}")
                    
                    if target_aligned.nunique() > 1 and feature_aligned.nunique() > 1:
                        try:
                            correlation = target_aligned.corr(feature_aligned)
                            if abs(round(correlation, 2)) >= threshold:
                                df_results[word] = correlation
                        except Exception as e:
                            print(f"Error calculating correlation for {word}: {e}")
                            continue
                    else:
                        continue

            target_results[df_name] = dict(sorted(df_results.items(),
                                                  key=lambda x: abs(x[1]),
                                                  reverse=True))

        corr_results[target_col] = target_results

    return corr_results

def get_examples(data_storage: DataStorage, input_corrs: Dict[str, Dict[str, Dict[str, float]]], num_sent: int = 7) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """
    Function for creating a dictionary with example sentences for correlated words.

    Args:
        data_storage: DataStorage class object containing prepared data
            (must contain encoded NLP dataframe and NLP dataframe)
        input_corrs: Correlation dictionary (result of find_nlp_corrs).
        num_sent: Number of example sentences for each word. Default is 10.

    Returns:
        Dict[str, Dict[str, Dict[str, List[str]]]]: Dictionary in the format:
            {
                target_col: {
                    source_col: {
                        word: [sentences],
                        ...
                    },
                    ...
                },
                ...
            }
    """
    if not hasattr(data_storage, 'nlp_df') or not hasattr(data_storage, 'nlp_lemmed_df'):
        raise ValueError("DataStorage object must have 'nlp_df' and 'nlp_lemmed_df' attributes")

    input_df = data_storage.nlp_df
    lemmed_input_df = data_storage.nlp_lemmed_df

    examples_dict = {}

    for target_col, df_results in input_corrs.items():
        target_examples = {}

        for source_col, word_corrs in df_results.items():
            col_examples = {}

            #  поиск индексов
            for word in word_corrs.keys():
                mask = lemmed_input_df[source_col].str.contains(
                    rf'\b{re.escape(word)}\b', regex=True, na=False)
                word_indices = lemmed_input_df[source_col][mask].index

                # примеры
                sentences = input_df.loc[word_indices, source_col].dropna()
                sentences = sentences[
                    ~sentences.str.lower().isin(['nan', '-', '_', ''])
                ].unique().tolist()

                sentences = [s.strip() for s in sentences if len(s.strip()) > 1]

                col_examples[word] = sentences[:10]

            if col_examples:
                target_examples[source_col] = col_examples

        examples_dict[target_col] = target_examples

    return examples_dict

def plot_clusters(data_storage: DataStorage, n_cl: int) -> pd.DataFrame:
    """
    Function for clustering text vector representations in dataframe columns using K-means method.

    Applies K-means algorithm to each dataframe column separately.

    Args:
        data_storage: DataStorage class object containing prepared data
            (must contain vectorized NLP dataframe)
        n_cl: Number of clusters for K-means algorithm.

    Returns:
        DataFrame: DataFrame with cluster labels for each element of the original dataframe.
    """
    if not hasattr(data_storage, 'nlp_vec_df'):
        raise ValueError("DataStorage object must have 'nlp_vec_df' attribute")

    input_df = data_storage.nlp_vec_df

    cluster_df = pd.DataFrame(index=input_df.index)

    for col in input_df.columns:
        vectors = np.vstack(input_df[col].dropna())
        kmeans = KMeans(n_clusters=n_cl, random_state=42)
        clusters = kmeans.fit_predict(vectors)
        cluster_df[f"{col}_cluster"] = clusters

    return cluster_df

def bplot_clusters(data_storage: DataStorage, n_cl: int) -> pd.DataFrame:
    """
    Function for clustering text vector representations using Bisecting K-means method.

    Applies Bisecting K-means algorithm to each dataframe column separately.

    Args:
        data_storage: DataStorage class object containing prepared data
            (must contain vectorized NLP dataframe)
        n_cl: Number of clusters for Bisecting K-means algorithm.

    Returns:
        DataFrame: DataFrame with cluster labels for each element of the original dataframe.
    """

    if not hasattr(data_storage, 'nlp_vec_df'):
        raise ValueError("DataStorage object must have 'nlp_vec_df' attribute")

    input_df = data_storage.nlp_vec_df

    cluster_df = pd.DataFrame(index=input_df.index)

    for col in input_df.columns:
        vectors = np.vstack(input_df[col].dropna())
        kmeans = BisectingKMeans(n_clusters=n_cl, random_state=42)
        clusters = kmeans.fit_predict(vectors)
        cluster_df[f"{col}_cluster"] = clusters

    return cluster_df

def check_clusters(data_storage: DataStorage, cluster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function for evaluating clustering quality using Calinski-Harabasz and Silhouette metrics.

    Args:
        data_storage: DataStorage class object containing prepared data
            (must contain vectorized NLP dataframe)
        cluster_df: DataFrame with cluster labels.

    Returns:
        DataFrame: Clustering quality evaluation results for each column.
    """

    if not hasattr(data_storage, 'nlp_vec_df'):
        raise ValueError("DataStorage object must have 'nlp_vec_df' attribute")

    original_df = data_storage.nlp_vec_df

    results = []

    for cluster_col in cluster_df.columns:
        original_col = cluster_col.replace('_cluster', '')

        vectors = np.vstack(original_df[original_col].dropna())
        labels = cluster_df[cluster_col].dropna()

        common_idx = labels.index.intersection(original_df[original_col].dropna().index)
        vectors = vectors[common_idx]
        labels = labels[common_idx]

        ch_score = calinski_harabasz_score(vectors, labels)
        sil_score = silhouette_score(vectors, labels)

        results.append({
            'Column': original_col,
            'Calinski-Harabasz': ch_score,
            'Silhouette': sil_score,
            'n_clusters': len(labels.unique())
        })

    return pd.DataFrame(results)

def get_cluster_examples(data_storage: DataStorage, cluster_df: pd.DataFrame, n_examples: int = 5) -> Dict[str, Dict[str, List[str]]]:
    """
    Function for retrieving example texts from each cluster.

    For each cluster in each column, selects the specified number of text examples.

    Args:
        data_storage: DataStorage class object containing prepared data
            (must contain NLP dataframe)
        cluster_df: DataFrame with cluster labels. Result of plot_clusters and bplot_clusters functions
        n_examples: Number of examples for each cluster. Default is 10.

    Returns:
        Dict[str, Dict[str, List[str]]]: Dictionary with text examples for each cluster.
    """

    if not hasattr(data_storage, 'nlp_df'):
        raise ValueError("DataStorage object must have 'nlp_df' attribute")

    input_df = data_storage.nlp_df

    cluster_examples = {}

    for cluster_col in cluster_df.columns:
        original_col = cluster_col.replace("_cluster", "")
        cluster_examples[original_col] = {}

        temp_df = pd.DataFrame({
            'text': input_df[original_col].astype(str),
            'cluster': cluster_df[cluster_col]
        }).dropna()

        temp_df = temp_df[
            ~temp_df['text'].str.lower().isin(['nan', '-', '_', ''])
        ].dropna()

        temp_df = temp_df[temp_df['text'].str.strip().astype(bool)]
        temp_df = temp_df.drop_duplicates(subset=['text', 'cluster'])

        for cluster_num, group in temp_df.groupby('cluster'):
            examples = group['text'].unique().tolist()
            examples = [ex.strip() for ex in examples if isinstance(ex, str) and len(ex.strip()) > 1]

            cluster_key = f"{cluster_col}_{cluster_num}"
            cluster_examples[original_col][cluster_key] = examples[:n_examples]

    return cluster_examples

def find_best_clusters(data_storage: DataStorage, min_ncl: int = 2, max_ncl: int = 6) -> pd.DataFrame:
    """
    Function for finding the best clustering among different methods and parameters.
    Compares clustering quality of K-means and Bisecting K-means for different numbers of clusters.

    Args:
        data_storage: DataStorage class object containing prepared data
            (must contain vectorized NLP dataframe)
        min_ncl: Minimum number of clusters to test. Default is 3.
        max_ncl: Maximum number of clusters to test. Default is 6.

    Returns:
        DataFrame: DataFrame with the best cluster labels for each column.
    """

    if not hasattr(data_storage, 'nlp_vec_df'):
        raise ValueError("DataStorage object must have 'nlp_vec_df' attribute")

    vector_df = data_storage.nlp_vec_df
    curr_coeff = 'Calinski-Harabasz'
    result_df = pd.DataFrame()

    for ncl in range(min_ncl, max_ncl + 1):
        kmeans_clusters = plot_clusters(data_storage, ncl)
        bkmeans_clusters = bplot_clusters(data_storage, ncl)

        kmeans_metrics = check_clusters(data_storage, kmeans_clusters)
        bkmeans_metrics = check_clusters(data_storage, bkmeans_clusters)

        for i, col in enumerate(kmeans_clusters.columns):
            kmeans_score = kmeans_metrics.iloc[i][curr_coeff]
            bkmeans_score = bkmeans_metrics.iloc[i][curr_coeff]

            if col not in result_df:
                if kmeans_score > bkmeans_score:
                    result_df[col] = kmeans_clusters[col]
                else:
                    result_df[col] = bkmeans_clusters[col]
            else:
                current_score = check_clusters(data_storage, pd.DataFrame({col: result_df[col]}))[curr_coeff].iloc[0]
                if kmeans_score > current_score or bkmeans_score > current_score:
                    if kmeans_score > bkmeans_score:
                        result_df[col] = kmeans_clusters[col]
                    else:
                        result_df[col] = bkmeans_clusters[col]

    return result_df

def late_nlp_examples(data_storage: DataStorage, bad_values: List[Union[str, float]] = ['', '-', '_', np.nan]) -> pd.DataFrame:
    """
    Функция для получения 10 непустых значений из каждого столбца DataFrame.

    Заменяет указанные плохие значения на пустые строки и возвращает по 10 непустых значений из каждого столбца.

    Args:
        data_storage: DataStorage object containing the DataFrame.
        bad_values: Список значений, которые считаются плохими и подлежат замене.
                   По умолчанию ['', '-', '_', np.nan].

    Returns:
        DataFrame: DataFrame с 10 непустыми значениями из каждого столбца.
    """

    if not hasattr(data_storage, 'late_nlp_df'):
        raise ValueError("DataStorage object must have 'late_nlp_df' attribute")
    
    df = data_storage.late_nlp_df

    columns = df.columns.tolist()

    def is_bad_value(x):
        if pd.isna(x):
            return ''
        if isinstance(x, str) and x.strip() in bad_values:
            return ''
        return x

    cleaned_df = df.copy()
    for col in columns:
        cleaned_df[col] = cleaned_df[col].apply(is_bad_value)
        
    result_df = pd.DataFrame()
    
    for col in columns:
        non_empty_values = cleaned_df[col][cleaned_df[col] != '']
        sample_values = non_empty_values.head(10)
        result_df[col] = sample_values.reset_index(drop=True)
    
    return result_df

def create_tools(data_storage, model, base_url, api_key, path_to_file):
    @tool
    def table_analysis_tool(query: str) -> str:
        """Provides information about the available HR data structure and columns.
        
        Args:
            query: Query about the data structure, columns, or available data
        """
        info = ""
        if hasattr(data_storage, 'orig_df') and not data_storage.orig_df.empty:
            info += "\nAvailable Columns:\n"
            for col in data_storage.orig_df.columns:
                info += f"- {col}: {data_storage.orig_df[col].dtype}\n"
        write_text_to_file("Table info:\n" + str(info), path_to_file)
        return info

    @tool
    def find_correlations_tool() -> Dict[str, Dict[str, float]]:
        """Find correlations in the data between target column and other columns"""
        result = find_correlations(data_storage)
        write_text_to_file("\nCorrelations with target column:\n" + str(result), path_to_file)
        return result

    @tool
    def count_answers_corr_tool() -> pd.Series:
        """Analyze answer correlations in the data and search whether final mark depend on answers count"""
        result = count_answers_corr(data_storage)
        write_text_to_file("\nCorrelations of count answers in column with target column:\n" + str(result), path_to_file)
        return result

    @tool
    def all_show_heatmap_tool() -> Dict[Tuple[str, str], float]:
        """Find correlations in the data between different answers. The answers are written under "_" after column names"""
        result = all_show_heatmap(data_storage)
        write_text_to_file("\nCorrelations between other columns, except target:\n" + str(result), path_to_file)
        return result

    @tool
    def find_nlp_corrs_tool() -> Dict[str, Dict[str, Dict[str, float]]]:
        """Find correlations in reviews which people wrote themselves (NLP). More precisely find correlations between target column and words in answers"""
        result = find_nlp_corrs(data_storage)
        write_text_to_file("\nCorrelations of words in reviews and target column\n" + str(result), path_to_file)
        return result

    @tool
    def get_examples_tool(corrs: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        """Get samples of NLP correlations from the data. Returns words and examples with them.
            Args:
                corrs: Dict[str, Dict[str, Dict[str, float]]] - result of find_nlp_corrs_tool
            Returns:
                res: Dict[str, Dict[str, Dict[str, List[str]]]] - sentencies which have correlation words
            
        """
        res = get_examples(data_storage, corrs)
        write_text_to_file("\nSanples of NLP correlations:\n" + str(res), path_to_file)
        return res

    @tool
    def find_best_clusters_tool() -> pd.DataFrame:
        """Find best clusters of given data and returns dataframe with clusters in NLP part of the data"""
        result = find_best_clusters(data_storage)
        write_text_to_file("\nAll clusters:\n" + str(result.dropna()), path_to_file)
        return result

    @tool
    def get_cluster_examples_tool(cluster_df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
        """Give sentence examples to each cluster
            Args:
                cluster_df: pd.DataFrame - dataframe with clusters. Result of find_best_clusters_tool
            Returns:
                result: Dict[str, Dict[str, List[str]]] - cluster examples
        """
        result = get_cluster_examples(data_storage, cluster_df)
        write_text_to_file("\nCluster examples:\n" + str(result), path_to_file)
        return result

    @tool
    def get_other_examples() -> str:
        """Get examples from columns with few answers"""
        result = late_nlp_examples(data_storage)
    
        non_empty_answers = []
    
        for column in result.columns:
            non_empty = result[column].dropna()
            non_empty = non_empty[non_empty.astype(str) != '']
            non_empty = non_empty[non_empty.astype(str) != 'nan']
            non_empty = non_empty[non_empty.astype(str) != 'NaN']
        
            if len(non_empty) > 0:
                non_empty_answers.append(f"\nКолонка: {column}")
                for i, answer in enumerate(non_empty.head(10), 1):
                    non_empty_answers.append(f"{i}. {answer}")
    
        answers_text = "\n".join(non_empty_answers)
    
        write_text_to_file("\nExamples from columns with few answers:\n" + answers_text + "\nEnd of examples", path_to_file)
    
        return str(answers_text)

    @tool
    def get_large_reviews() -> str:
        """Get all large NLP reviews"""
        result = data_storage.nlp_large_df

        non_empty_answers = []

        for column in result.columns:
            non_empty = result[column].dropna()
            non_empty = non_empty[non_empty.astype(str) != '']
            non_empty = non_empty[non_empty.astype(str) != 'nan']
            non_empty = non_empty[non_empty.astype(str) != 'NaN']

            if len(non_empty) > 0:
                non_empty_answers.append(f"\nКолонка: {column}")
                for i, answer in enumerate(non_empty, 1):
                    non_empty_answers.append(f"{i}. {answer}")

        answers_text = "\n".join(non_empty_answers)

        write_text_to_file("\nExamples of large answers:\n" + str(answers_text) + "\nEnd of examples", path_to_file)
        return str(answers_text)
    
    @tool
    def get_critic_answer(report: str) -> str:
        """Get critics for report.
            Args:
                report: agent-made report
            Returns:
                critics: critic's answer
        """
        path = path_to_file + '/tmp.txt'
        with open(path, 'r', encoding='utf-8') as file:
            existing_content = file.read()
        critics = generate_critics(model, base_url, api_key, report, existing_content)
        return critics

    @tool
    def get_tools_results() -> str:
        "Get results of all tools"
        path = path_to_file + '/tmp.txt'
        with open(path, 'r', encoding='utf-8') as file:
            existing_content = file.read()
        return existing_content
    
    return [table_analysis_tool,
        find_correlations_tool,
        count_answers_corr_tool,
        all_show_heatmap_tool,
        find_nlp_corrs_tool,
        get_examples_tool,
        find_best_clusters_tool,
        get_cluster_examples_tool,
        get_other_examples,
        get_large_reviews,
        get_critic_answer,
        FinalAnswerTool()]
