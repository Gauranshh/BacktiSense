# 🦠 BactiSense — AI Antibiotic Intelligence System
### CodeCure Biohackathon · Track B: Antibiotic Resistance Prediction

## Problem
Antimicrobial resistance (AMR) is killing ~700,000 people/year globally.
Clinicians need fast, data-driven guidance on which antibiotics will work.

## Solution
BactiSense uses a Random Forest classifier trained on bacterial isolate 
susceptibility data to predict resistance patterns and recommend optimal 
treatment per sampling location.

## Setup & Run
pip install -r requirements.txt
python python.py        # trains model, exports JSON
streamlit run app.py    # launches UI

## Results
| Antibiotic     | Accuracy |
|----------------|----------|
| IMIPENEM       | 83.64%   |
| CEFTAZIDIME    | 78.18%   |
| GENTAMICIN     | 60.0%    |
| AUGMENTIN      | 81.82%   |
| CIPROFLOXACIN  | 67.27%   |

## Dataset
Primary: Antimicrobial Resistance Dataset (Mendeley, DOI: 10.17632/ccmrx8n7mk.1)

## Track B Deliverables
✅ GitHub Repository  
✅ Resistance prediction model  
✅ Location Influence Analysis 
✅ Resistance Pattern Network visualization  
✅ Decision-support tool suggesting antibiotics