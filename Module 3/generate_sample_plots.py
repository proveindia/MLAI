import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")

# Create images directory if it doesn't exist
output_dir = "images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load dataset
df = sns.load_dataset('tips')

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=100)
    plt.close()
    print(f"Saved {filename}")

# 1. Histogram
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='total_bill', kde=True)
plt.title("Histogram of Total Bill")
save_plot("histogram.png")

# 2. KDE Plot
plt.figure(figsize=(8, 6))
sns.kdeplot(data=df, x='total_bill', fill=True)
plt.title("KDE of Total Bill")
save_plot("kde.png")

# 3. Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='day', y='total_bill')
plt.title("Boxplot of Total Bill by Day")
save_plot("boxplot.png")

# 4. Countplot
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='day')
plt.title("Count of Tips by Day")
save_plot("countplot.png")

# 5. Scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='total_bill', y='tip', hue='time')
plt.title("Scatterplot of Tip vs Total Bill")
save_plot("scatterplot.png")

# 6. Barplot
plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='sex', y='total_bill', estimator='mean')
plt.title("Average Total Bill by Sex")
save_plot("barplot.png")

# 7. Heatmap
plt.figure(figsize=(8, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
save_plot("heatmap.png")

# 8. Pairplot
# Pairplot handles its own figure
g = sns.pairplot(df, hue='sex')
g.fig.suptitle("Pairplot of Tips Dataset", y=1.02)
g.savefig(os.path.join(output_dir, "pairplot.png"), dpi=100)
plt.close()
print("Saved pairplot.png")

# 9. Violin Plot
plt.figure(figsize=(8, 6))
sns.violinplot(data=df, x="day", y="total_bill")
plt.title("Violin Plot of Total Bill by Day")
save_plot("violinplot.png")
