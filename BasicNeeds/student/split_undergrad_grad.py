import pandas as pd
from datetime import datetime

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

# Filter to keep ONLY these student roles for student data
student_data = data[data['Type'].isin(student_roles)]

# Get non-student data
non_student_data = data[~data['Type'].isin(student_roles)]

# Get 'Other' responses
other_responses = data[data['Type'] == 'Other (please specify)']
# Get the text from the 'Type_other' column for these responses
if 'Type_other' in data.columns:
    other_details = other_responses['Type_other'].value_counts()

# Count all types including non-student roles
all_type_counts = data['Type'].value_counts()

# Count student responses
total_students = len(student_data)
undergrad_ft_count = len(student_data[student_data['Type'] == 'Undergraduate student (Full-time, 12 credit hours or more)'])
undergrad_pt_count = len(student_data[student_data['Type'] == 'Undergraduate student (Part-time, less than 12 credit hours)'])
grad_count = len(student_data[student_data['Type'] == 'Graduate or professional student'])

# Generate timestamp for the filename
output_file = f'all_role_counts.txt'

# Write results to file
with open(output_file, 'w') as f:
    f.write("\nComplete breakdown of all types in dataset:\n")
    f.write("-------------------------------------------\n")
    for type_name, count in all_type_counts.items():
        f.write(f"{type_name}: {count}\n")
    
    f.write("\nSummary:\n")
    f.write("--------\n")
    f.write(f"Total students: {total_students}\n")
    f.write(f"- Full-time undergrad students: {undergrad_ft_count}\n")
    f.write(f"- Part-time undergrad students: {undergrad_pt_count}\n")
    f.write(f"- Graduate students: {grad_count}\n")
    f.write(f"\nTotal non-student responses: {len(non_student_data)}\n")
    
    # Add Other responses breakdown
    f.write(f"\nBreakdown of 'Other' responses ({len(other_responses)} total):\n")
    f.write("----------------------------------------\n")
    if 'Type_other' in data.columns:
        for response, count in other_details.items():
            if pd.notna(response):  # Only write non-NaN responses
                f.write(f"{response}: {count}\n")
    else:
        f.write("No 'Type_other' column found in the dataset\n")

print(f"Results saved to {output_file}")