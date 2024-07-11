import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
mtcars = pd.read_csv("mtcars.csv")

# Plot histogram
plt.hist(mtcars['mpg'], bins=10, color='blue', edgecolor='black')

# Add labels and title
plt.xlabel('Miles per gallon (mpg)')
plt.ylabel('Frequency')
plt.title('Histogram of Miles per gallon (mpg)')

# Show plot
plt.show()
