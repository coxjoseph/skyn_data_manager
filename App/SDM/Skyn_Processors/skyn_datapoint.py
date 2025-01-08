import numpy as np
import statistics
from scipy import optimize


class SkynDatapoint:
    def __init__(self, column_name, datetime_column, i, df, extension_range, sampling_rate, use_before=True,
                 use_after=True, label=None):
        self.column_name = column_name
        self.datetime_column = datetime_column
        self.index = i
        self.data = df
        self.value = self.data.loc[self.index, self.column_name]
        self.current_time = self.data.loc[self.index, self.datetime_column]
        self.label = label

        self.extension_range = extension_range  # how many values to consider before and after current value

        if self.index - 1 in self.data.index and use_before:
            self.value_before = self.data.loc[self.index - 1, self.column_name]
            self.difference_from_prior = (self.value - self.value_before)
            self.values_before = self.get_prior_values()
            self.prior_n = len(self.values_before)
            self.duration_before = len(self.values_before) * sampling_rate
            self.mean_before = sum(self.values_before) / len(self.values_before)
            self.mean_change_before = self.value - self.mean_before
            self.std_before = statistics.stdev(self.values_before) if self.prior_n > 1 else np.nan
            self.duration_minutes_before = (self.current_time - self.data.loc[
                self.index - len(self.values_before), datetime_column]).total_seconds() / 60
            self.x_time_before = [self.duration_minutes_before * (i - len(self.values_before)) / len(self.values_before)
                                  for i in range(0, len(self.values_before))]
            self.a_pre, self.b_pre, self.c_pre = self.get_quadratic_params(self.x_time_before + [0],
                                                                           self.values_before + [self.value])

        if self.index + 1 in self.data.index and use_after:
            self.value_after = self.data.loc[self.index + 1, self.column_name]
            self.difference_from_next = self.value_after - self.value
            self.values_after = self.get_post_values()
            self.after_n = len(self.values_after)
            self.duration_after = len(self.values_after) * sampling_rate
            self.mean_after = sum(self.values_after) / len(self.values_after)
            self.mean_change_after = self.value - self.mean_after
            self.std_after = statistics.stdev(self.values_after) if self.after_n > 1 else np.nan
            self.duration_minutes_after = (self.data.loc[self.index + len(
                self.values_after), datetime_column] - self.current_time).total_seconds() / 60
            self.x_time_after = [self.duration_minutes_after * i / len(self.values_after) for i in
                                 range(1, len(self.values_after) + 1)]
            self.a_post, self.b_post, self.c_post = self.get_quadratic_params([0] + self.x_time_after,
                                                                              [self.value] + self.values_after)

        if (self.index - 1 in self.data.index and use_before) and (self.index + 1 in self.data.index and use_after):
            self.values_before_after = self.values_before + [self.value] + self.values_after
            self.n_total = len(self.values_before_after)
            self.a_all, self.b_all, self.c_all = self.get_quadratic_params(self.x_time_before + [0] + self.x_time_after,
                                                                           self.values_before_after)

    def get_prior_values(self):
        before = []
        for i in range(1, self.extension_range + 1):
            if self.index - i in self.data.index.tolist():
                before.append(self.data.loc[self.index - i, self.column_name])
        return before

    def get_post_values(self):
        after = []
        for i in range(1, self.extension_range + 1):
            if self.index + i in self.data.index.tolist():
                after.append(self.data.loc[self.index + i, self.column_name])
        return after

    @staticmethod
    def quadratic_model(x, a, b, c):
        return a * x ** 2 + b * x + c

    # noinspection PyTupleAssignmentBalance
    def get_quadratic_params(self, x, y):
        if len(y) <= 5:
            return np.nan, np.nan, np.nan
        try:
            params, _ = optimize.curve_fit(self.quadratic_model, np.array(x), np.array(y))
            return tuple(params[:3])
        except (RuntimeError, ValueError) as e:
            print(f"Error fitting quadratic model: {e}")
            return np.nan, np.nan, np.nan
