import pandas as pd

# Sample DataFrame
data = {'Column1': [1, 2, 3], 'Column2': ['A', 'B', 'C']}
df = pd.DataFrame(data)

# Excel file path (if it doesn't exist, it will be created)
excel_file_path = 'example.xlsx'

# Save the DataFrame to Excel
df.to_excel(excel_file_path, index=False)
print(f'Data saved to {excel_file_path}')