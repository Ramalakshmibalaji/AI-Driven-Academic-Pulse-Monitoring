import pandas as pd

# Load cleaned publication dataset
file_path = "cleaned_publication_dataset.csv"
df = pd.read_csv(file_path)

# Create consistent anonymized course mapping
course_mapping = {
    course: f"Course {i+1}"
    for i, course in enumerate(sorted(df["course_name"].dropna().unique()))
}

# Replace original course names
df["course_name"] = df["course_name"].map(course_mapping)

# Save anonymized publication dataset
df.to_csv("cleaned_publication_dataset.csv", index=False)

print("Course names anonymized successfully!")
print("\nCourse Mapping:")
for original, anonymized in course_mapping.items():
    print(f"{original} -> {anonymized}")

print("\nDataset shape:", df.shape)
print("\nAnonymized courses:")
print(sorted(df["course_name"].dropna().unique()))