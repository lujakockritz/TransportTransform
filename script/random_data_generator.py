import pandas as pd
import random

number_random_agents = 100

# Define the possible values for each column
ages = list(range(16, 100))
sexes = ['female', 'male']
frequencies = ['(Almost) daily', '1-3 days a week', '1-3 days a month', 'Less often than monthly', 'Never or almost never']
preferences = ['d_public', 'd_carsh', 'd_car', 'd_bike']
license_values = [True, False]

# %%
# Generate the dummy dataset
data = []
for i in range(0, number_random_agents):
    row = {
        'ID': i,
        'ID.1': i,
        'Original_ID': i,
        'group': random.randint(1, 10),
        'c_age': random.choice(ages),
        'c_sex': random.choice(sexes),
        'c_income': random.choice(frequencies),
        'c_education': random.choice(frequencies),
        'c_occupation': random.choice(frequencies),
        'f_car_petrol': random.choice(frequencies),
        'f_car_passenger': random.choice(frequencies),
        'f_carsh': random.choice(frequencies),
        'f_public': random.choice(frequencies),
        'f_bike': random.choice(frequencies),
        'Attitude_ICECar': round(random.uniform(1, 6), 2),
        'Attitude_PT': round(random.uniform(1, 6), 2),
        'Attitude_Bike': round(random.uniform(1, 6), 2),
        'Attitude_CarSh': round(random.uniform(1, 6), 2),
        'preference_1': random.choice(preferences),
        'preference_2': random.choice(preferences),
        'license': random.choice(license_values),
        'car_owned': random.choice(license_values)
    }
    data.append(row)

# Create the DataFrame
df = pd.DataFrame(data)

# %%
# Save the DataFrame to a CSV file
df.to_csv('dummy_dataset.csv', index=False)