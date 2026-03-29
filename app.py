import os
import sys
import joblib
from narwhals import col
import pandas as pd
import numpy as np
from fastapi import FastAPI,File,UploadFile,Request
from catboost import Pool
from fastapi import Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse,RedirectResponse
from fastapi.staticfiles import StaticFiles
import traceback
from collections import defaultdict
app=FastAPI()


templates=Jinja2Templates(directory="./templates")
app.mount("/static",StaticFiles(directory="static"),name="static")

model=joblib.load("final_model/model.pkl")
cat_cols=joblib.load("final_model/categorical_columns.pkl")
cols=joblib.load("final_model/feature_names.pkl")
cat_indices=joblib.load("final_model/cat_indices.pkl")
cat_values=joblib.load("final_model/categorical_values.pkl")

def group_columns(cols):
    sections=defaultdict(list)
    for col in cols:
        sections[detect_section(col)].append(col)
    return sections

def detect_placeholder(cols):
    c=cols.lower()
    if "age" in c: return "e.g. 25"
    if "bpm" in c: return "e.g. 140"
    if "water" in c: return "liters per day"
    if "calorie" in c: return "kcal"
    if "duration" in c: return "hours"
    if "frequency" in c: return "0-7"
    if "bmi" in c: return "e.g. 22.5"
    if "physical" in c: return "intensity (0-4)"
    return "Enter Value"

def detect_section(cols):
    col_lower=cols.lower()
    if any(k in col_lower for k in ["age","gender","bmi"]):
        return "Personal Info"

    if any(k in col_lower for k in ["bpm","heart"]):
        return "Heart Data"

    if any(k in col_lower for k in ["workout","session","sets","exercise","reps"]):
        return "Workout"

    if any(k in col_lower for k in ["calorie","diet","water","meal"]):
        return "Nutrition"

def generate_form_fields(cols, cat_cols, cat_values):
    sections = group_columns(cols)
    html = ""
    for section, section_cols in sections.items():
        html += f"""
        <div class="section-card">
            <div class="section-top">
                <h2 class="section-header">{section}</h2>
                <div class="section-divider"></div>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        """
        for col in section_cols:
            placeholder = detect_placeholder(col)
            if col in cat_cols:
                options = "".join(
                    [f'<option value="{v}">{v}</option>'
                     for v in cat_values.get(col, [])]
                )

                field = f"""
                <div class="input-group">
                    <label>{col}</label>
                    <select name="{col}" required>
                        {options}
                    </select>
                </div>
                """
            else:
                field = f"""
                <div class="input-group">
                    <label>{col}</label>
                    <input type="number" step="any"
                        name="{col}"
                        placeholder="{placeholder}"
                        required>
                </div>
                """
            html += field
        html += "</div></div>"
    return html

@app.get("/")
async def home():
    return RedirectResponse("/app")

@app.get("/app")
async def app_ui(request: Request):
    form_html=generate_form_fields(cols,cat_cols,cat_values)   
    return templates.TemplateResponse("index.html",{"request":request,"form_html":form_html})

@app.post("/predict_form",response_class=HTMLResponse)
async def predict_form(request:Request):
    form=await request.form()
    form_data=dict(form)
    try:
        df=pd.DataFrame([form_data])
        df=df[cols]
        for col in cat_cols:
            df[col]=df[col].astype(str)
        
        for col in cols:
            if col not in cat_cols:
                df[col]=pd.to_numeric(df[col])

        pool=Pool(df,cat_features=cat_indices)
        pred=model.predict(pool).flatten()[0]

        if pred=="Low":
            interpretation="Low Fitness Level"
            recommendation="Increase physical activity, start with light cardio, and build a consistent workout routine."
        elif pred=="Medium":
            interpretation="Moderate Fitness Level"
            recommendation="Maintain your routine and gradually include strength training and endurance workouts."
        else: 
            interpretation="High Fitness Level"
            recommendation="Great job! Maintain your fitness and consider optimizing performance with advanced training."
        return templates.TemplateResponse("result.html",{"request":request,"result":pred,"interpretation":interpretation,"recommendation":recommendation})
    except Exception as e:
        traceback.print_exc()
        return HTMLResponse(f"<h2>Error:</h2><pre>{str(e)}</pre>")

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000) 