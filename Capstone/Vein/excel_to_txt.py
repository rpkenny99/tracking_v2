import pandas as pd

def extract_and_save(filename, output_txt):
    # Load the Excel file
    df = pd.read_excel(filename)
    
    # Extract the required columns
    if {'Tx', 'Ty', 'Tz'}.issubset(df.columns):
        df_selected = df[['Tx', 'Ty', 'Tz']]
    else:
        raise ValueError("The required columns (Tx, Ty, Tz) are not found in the Excel file.")
    
    # Format and save to a text file
    with open(output_txt, 'w') as f:
        for row in df_selected.itertuples(index=False, name=None):
            f.write(f"{row[0]} {row[1]} {row[2]}\n")
    
    print(f"File saved successfully as {output_txt}")

# Run the function with the specified filename and output file
extract_and_save("Capstone/Vein/rightvein2.xlsx", "right_vein.txt")