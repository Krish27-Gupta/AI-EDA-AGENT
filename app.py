            with open('file_loader.py', 'w') as f:
                f.write(code[6:])
    except Exception as e:
        # Fallback file loader template if LLM code extraction fails
        fallback_code = """
import pandas as pd
def read_uploaded_file(path):
    if str(path).endswith('.csv'):
        return pd.read_csv(path)
    elif str(path).endswith(('.xlsx', '.xls')):
        return pd.read_excel(path)
    else:
        try:
            return pd.read_csv(path)
        except:
            return pd.read_excel(path)
"""
        with open('file_loader.py', 'w') as f:
            f.write(fallback_code)

    return "Success"

def read_file(path):
    from file_loader import read_uploaded_file
    return read_uploaded_file(path)

# ==========================================
# EDA GENERATION FUNCTIONS
# ==========================================
def perform_eda_func(data, agent_inst):
    df = data.sample(min(len(data), 5))
    prompt = f"""You are a data analysts perform basic eda python single function perform_eda code and give all required analysis like missing values and columns Data frame sample : {df} data stats: {df.describe()}"""

    response = agent_inst.invoke({'messages':[{'role':'user','content':prompt}]})
    ans = response["messages"][-1].content[-1]['text']
    code = ans.split("