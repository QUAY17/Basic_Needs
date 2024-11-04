
import pandas as pd

# Load the CSV file
input_file = 'data/student/statewide_student_23.csv'  

# Read the CSV file into a DataFrame
data = pd.read_csv(input_file, low_memory=False)

# Define the student roles we want to keep
student_roles = [
    'Undergraduate student (Part-time, less than 12 credit hours)',
    'Undergraduate student (Full-time, 12 credit hours or more)',
    'Graduate or professional student',
]

# Filter to keep ONLY these student roles
student_data = data[data['Type'].isin(student_roles)]

# Save the combined student data
student_output_file = 'all_students_data.csv'
student_data.to_csv(student_output_file, index=False)

# Still create separate files for each student type if needed
undergrad_pt_data = student_data[student_data['Type'] == 'Undergraduate student (Part-time, less than 12 credit hours)']
undergrad_ft_data = student_data[student_data['Type'] == 'Undergraduate student (Full-time, 12 credit hours or more)']
grad_data = student_data[student_data['Type'] == 'Graduate or professional student']

# Save individual files
undergrad_pt_file = 'undergrad_pt_data.csv'
undergrad_ft_file = 'undergrad_ft_data.csv'
grad_file = 'grad_data.csv'

undergrad_pt_data.to_csv(undergrad_pt_file, index=False)
undergrad_ft_data.to_csv(undergrad_ft_file, index=False)
grad_data.to_csv(grad_file, index=False)

# Count responses
total_students = len(student_data)
undergrad_ft_count = len(undergrad_ft_data)
undergrad_pt_count = len(undergrad_pt_data)
grad_count = len(grad_data)

# Print results
print(f"Combined student data saved to {student_output_file} with {total_students} total students")
print(f"Breakdown:")
print(f"- Full-time undergrad students: {undergrad_ft_count}")
print(f"- Part-time undergrad students: {undergrad_pt_count}")
print(f"- Graduate students: {grad_count}")