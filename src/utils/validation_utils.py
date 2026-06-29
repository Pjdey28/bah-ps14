def check_columns(df,required):

    missing=[]

    for col in required:

        if col not in df.columns:
            missing.append(col)

    return missing