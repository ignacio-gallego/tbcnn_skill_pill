from optimus.helpers.columns import check_column_numbers
from optimus.helpers.columns import parse_columns
from optimus.plots.functions import plot_scatterplot, plot_boxplot, plot_frequency, plot_hist, \
    plot_correlation, plot_qqplot


class Plot:
    def __init__(self, df):
        self.df = df

    def hist(self, columns=None, buckets=10, output_format="plot", output_path=None):
        """
        Plot histogram
        :param columns: Columns to be printed
        :param buckets: Number of buckets
        :param output_format:
        :param output_path: path where the image is going to be saved
        :return:
        """
        df = self.df
        columns = parse_columns(df, columns)

        data = df.cols.hist(columns, buckets)["hist"]
        for col_name in data.keys():

            plot_hist({col_name: data[col_name]}, output=output_format, path=output_path)

    def scatter(self, columns=None, buckets=30, output_format="plot", output_path=None):
        """
        Plot scatter
        :param columns: columns to be printed
        :param buckets: number of buckets
        :param output_format:
        :param output_path: path where the image is going to be saved
        :return:
        """
        df = self.df
        columns = parse_columns(df, columns, filter_by_column_dtypes=df.constants.NUMERIC_TYPES)
        check_column_numbers(columns, "*")

        data = df.cols.scatter(columns, buckets)
        plot_scatterplot(data, output=output_format, path=output_path)

    def box(self, columns=None, output_format="plot", output_path=None):
        """
        Plot boxplot
        :param columns: Columns to be printed
        :param output_format:
        :param output_path: path where the image is going to be saved
        :return:
        """
        df = self.df
        columns = parse_columns(df, columns, filter_by_column_dtypes=df.constants.NUMERIC_TYPES)
        check_column_numbers(columns, "*")

        for col_name in columns:
            stats = df.cols.boxplot(col_name)
            plot_boxplot({col_name: stats}, output=output_format, path=output_path)

    def frequency(self, columns=None, buckets=10, output_format="plot", output_path=None):
        """
        Plot frequency chart
        :param columns: Columns to be printed
        :param buckets: Number of buckets
        :param output_format:
        :param output_path: path where the image is going to be saved
        :return:
        """
        df = self.df
        columns = parse_columns(df, columns)
        data = df.cols.frequency(columns, buckets)

        for k, v in data.items():
            plot_frequency({k: v}, output=output_format, path=output_path)

    def correlation(self, col_name, method="pearson", output_format="plot", output_path=None):
        """
        Compute the correlation matrix for the input data set of Vectors using the specified method. Method
        mapped from pyspark.ml.stat.Correlation.
        :param col_name: The name of the column for which the correlation coefficient needs to be computed.
        :param method: String specifying the method to use for computing correlation. Supported: pearson (default),
        spearman.
        :param output_format: Output image format
        :param output_path: Output path
        :return: Heatmap plot of the corr matrix using seaborn.
        """
        df = self.df
        cols_data = df.cols.correlation(col_name, method, output="array")
        plot_correlation(cols_data, output=output_format, path=output_path)

    def qqplot(self, columns, n=100, output_format="plot", output_path=None):
        """
        QQ plot
        :param columns:
        :param n: Sample size
        :param output_format: Output format
        :param output_path: Path to the output file
        :return:
        """
        df = self.df

        columns = parse_columns(df, cols_args=columns, filter_by_column_dtypes=df.constants.NUMERIC_TYPES)

        if columns is not None:
            sample_data = df.sample(n=n, random=True)
            for col_name in columns:
                plot_qqplot(col_name, sample_data, output=output_format, path=output_path)
