import pandas as pd
import matplotlib.pyplot as plt


def analyze_train_set():
    # df = pd.read_csv("/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/data_reader/dataset/training_data_info.csv")
    df = pd.read_csv("/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/data_reader/dataset/test_data_info.csv")

    #
    # Age:
    #
    df_age = df['age']
    df_age = df_age.dropna()
    print(len(df_age))
    hist = df_age.hist(bins=100)
    plt.legend()
    plt.title("Age Histogram")
    plt.show()
    print(f"Age Max: {df_age.max()}")
    print(f"Age Min: {df_age.min()}")
    print(f"Age mean: {df_age.mean()}")

    #
    # Gender:
    #
    plt.figure()
    df_gender = df['gender']
    print(df_gender.value_counts())
    hist = df_gender.hist(bins=3)
    plt.title("Gender")
    plt.show()

    #
    # Sampling rates:
    #
    print(df['samp_rates'].value_counts())
    print(df['duration'].value_counts())
    print(df['num_samples'].value_counts())

if __name__ == "__main__":
    analyze_train_set()