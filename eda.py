import matplotlib.pyplot as plt
import seaborn as sns


def plot_price_variation_per_sector(df):
    # Extract the columns we need in this step from the dataframe
    df_ = df.loc[:, ['Stock', 'Sector', 'PRICE VAR [%]']]

    # Get list of sectors
    sector_list = df_['Sector'].unique()

    # Plot the percent price variation for each sector
    for sector in sector_list:
        temp = df_[df_['Sector'] == sector]

        plt.figure(figsize=(30, 5))
        plt.plot(temp['Stock'], temp['PRICE VAR [%]'])
        plt.title(sector.upper(), fontsize=20)
        plt.show()


def plot_correlation_matrix(df):
    corrMatrix = df.corr()
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(corrMatrix, annot=False, cmap='YlGnBu', vmin=-1, vmax=1, center=0, ax=ax)
    plt.show()
