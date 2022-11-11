import matplotlib.pyplot as plt

fig, ax = plt.subplots()


def plotResults(xvalues, yvalues, title, ylabel):
    bar_labels = ['red', 'blue', 'orange']
    bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

    ax.bar(xvalues, yvalues, label=bar_labels)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # ax.legend(title)
    plt.grid()
    plt.show()

xMSE = ['CYLINDERS', 'FUELCONSUMPTION_COMB', 'ENGINESIZE']
yMSE = [1179.30, 984,  959.96]
yMAE = [26.37, 21.73, 23.57]
yR2 = [69, 77, 75]
titleMSE = "MSE Results"
titleMAE = "MAE Results"
titleR2 = "R2_score Results"
y_MSE = "MSE"
y_MAE = 'MAE'
y_R2 = 'R2_Score'

# plotResults(xMSE, yMSE, titleMSE, y_MSE)

plotResults(xMSE, yMAE, titleMAE, y_MAE)

# plotResults(xMSE, yR2, titleR2, y_R2)