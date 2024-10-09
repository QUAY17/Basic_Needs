
import pandas as pd

# Load the CSV file
input_file = 'data/statewide_facultystaff_24.csv'  

# Read the CSV file into a DataFrame
data = pd.read_csv(input_file)

# Define the roles to filter out
roles_to_filter = [
    'Staff (Part-time, Full-time, or temporary)',
    'Faculty (Part-time or Adjunct)',
    'Faculty (Full-time)',
    'Lecturer (Full-time)',
    'Upper-level administration'
]

# Filter out these roles
filtered_data = data[~data['Type'].isin(roles_to_filter)]

# Save the filtered DataFrame to a new CSV file
filtered_output_file = 'split_statewide_facultystaff.csv'
filtered_data.to_csv(filtered_output_file, index=False)

# Separate the filtered data into two categories: STAFF and FACULTY

# Define the roles for STAFF and FACULTY
staff_roles = ['Staff (Part-time, Full-time, or temporary)']
faculty_roles = [
    'Faculty (Part-time or Adjunct)',
    'Faculty (Full-time)',
    'Lecturer (Full-time)',
    'Upper-level administration'
]

# Filter out the data for each category
staff_data = data[data['Type'].isin(staff_roles)]
faculty_data = data[data['Type'].isin(faculty_roles)]

# Save both datasets into separate CSV files
staff_output_file = 'staff_data.csv'
faculty_output_file = 'faculty_data.csv'

staff_data.to_csv(staff_output_file, index=False)
faculty_data.to_csv(faculty_output_file, index=False)

# Differentiate staff based on column Employment
# Assuming that CX contains values like 'Part-time' and 'Full-time'
staff_part_time = staff_data[staff_data['Employment'] == 'Employed, part-time']
staff_full_time = staff_data[staff_data['Employment'] == 'Employed, full-time']

# Count the number of responses for each category and subtype
staff_count = len(staff_data)
faculty_count = len(faculty_data)

# Count individual subtypes for staff and faculty
staff_combined_count = len(data[data['Type'] == 'Staff (Part-time, Full-time, or temporary)'])
staff_part_time_count = len(staff_part_time)
staff_full_time_count = len(staff_full_time)

faculty_part_time_count = len(data[data['Type'] == 'Faculty (Part-time or Adjunct)'])
faculty_full_time_count = len(data[data['Type'] == 'Faculty (Full-time)'])
lecturer_full_time_count = len(data[data['Type'] == 'Lecturer (Full-time)'])
upper_admin_count = len(data[data['Type'] == 'Upper-level administration'])

# Save part-time and full-time staff data
staff_part_time_output_file = 'staff_part_time_data.csv'
staff_full_time_output_file = 'staff_full_time_data.csv'

staff_part_time.to_csv(staff_part_time_output_file, index=False)
staff_full_time.to_csv(staff_full_time_output_file, index=False)

# Print the results
print(f"Filtered data saved to {filtered_output_file}")
print(f"Staff data saved to {staff_output_file} with {staff_count} responses")
print(f"Faculty data saved to {faculty_output_file} with {faculty_count} responses")

print(f"Staff (Part-time, Full-time, or Temporary): {staff_combined_count}")
print(f"Staff Part-time: {staff_part_time_count}")
print(f"Staff Full-time: {staff_full_time_count}")

print(f"Faculty Part-time or Adjunct: {faculty_part_time_count}")
print(f"Faculty Full-time: {faculty_full_time_count}")
print(f"Lecturer Full-time: {lecturer_full_time_count}")
print(f"Upper-level administration: {upper_admin_count}")

# Calculating percentages
staff_part_time_percentage = (staff_part_time_count / staff_count) * 100
staff_full_time_percentage = (staff_full_time_count / staff_count) * 100

faculty_part_time_percentage = (faculty_part_time_count / faculty_count) * 100
faculty_full_time_percentage = (faculty_full_time_count / faculty_count) * 100
lecturer_full_time_percentage = (lecturer_full_time_count / faculty_count) * 100
upper_admin_percentage = (upper_admin_count / faculty_count) * 100

print(f"Staff Part-Time Percentage: {staff_part_time_percentage:.2f}%"),
print(f"Staff Full-Time Percentage: {staff_full_time_percentage:.2f}%"),
print(f"Faculty Part-Time Percentage: {faculty_part_time_percentage:.2f}%"),
print(f"Faculty Full-Time Percentage: {faculty_full_time_percentage:.2f}%"),
print(f"Lecturer Full-Time Percentage: {lecturer_full_time_percentage:.2f}%"),
print(f"Upper-Level Administration Percentage: {upper_admin_percentage:.2f}%")



