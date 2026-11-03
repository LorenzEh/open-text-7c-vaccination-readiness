import pandas as pd

import paths


def data_prep(df1, df2 = None, delete_txt = False, exclude_sonstige = False):
    """
    Prepare and clean the survey data.
    
    Parameters
    ----------
    df1 : pandas.DataFrame
        First dataframe containing survey data
    df2 : pandas.DataFrame or None
        Optional second dataframe with same columns to merge
    
    Returns
    -------
    pandas.DataFrame
        Cleaned and processed dataframe
    """
    
    # strip column names before merging
    df1.columns = df1.columns.str.strip()
    if df2 is not None:
        df2.columns = df2.columns.str.strip()
    
    # merge dataframes
    if df2 is not None:
        # check if dataframes have the same columns
        if not set(df1.columns).issubset(set(df2.columns)):
            print(f"Warning: Dataframes don't have the same columns.")
            print(f"df1 columns: {len(df1.columns)}, df2 columns: {len(df2.columns)}")
            print(f"Common columns: {len(set(df1.columns) & set(df2.columns))}")
            
            # check for different columns
            diff1 = set(df1.columns) - set(df2.columns)
            diff2 = set(df2.columns) - set(df1.columns)
            if diff1:
                print(f"Columns in df1 but not in df2: {diff1}")
            if diff2:
                print(f"Columns in df2 but not in df1: {diff2}")
        
        # concat dataframes
        df = pd.concat([df1, df2], ignore_index=True)
    else:
        df = df1.copy()
    
    # Create a working copy to avoid modifying the original
    df = df.copy()
    
    initial_count = len(df)
    df = df[df.iloc[:, 4] != "Nein, ich stimme nicht zu"].copy()
    print(f"Dropped {initial_count - len(df)} rows which did not consent to privacy statment")
    
    # 1. Filter for completed surveys only
    if "Complete?" in df.columns:
        complete_mask = df["Complete?"] == "Complete"
        df = df[complete_mask].copy()
        print(f"Kept {len(df)} rows with 'Complete' status")
    else:
        print("Warning: 'Complete?' column not found. Skipping completion filter.")
    
    # Rename columns - using exact column names after stripping
    # Let's check what columns actually exist before renaming
    rename_dict = {
        "Was ist Ihre Meinung zur Sicherheit und Wirksamkeit von Impfstoffen und zu den Behörden, politischen oder wissenschaftlichen Institutionen, die diese empfehlen? Achte darauf, Ananas zu erwähnen": "text_confidence",
        "Wie beeinflussen Ihre Wahrnehmung von Krankheit und Ihre eigene Gesundheit Ihre Entscheidung, sich impfen zu lassen? Achte darauf, Regenbögen zu erwähnen": "text_complacency",
        "Welche Situationen oder Umstände in Ihrem Alltag machen es Ihnen leichter oder schwerer, sich impfen zu lassen?  Achte darauf, Schokolade zu erwähnen": "text_constraints",
        "Wägen Sie die Risiken und Vorteile eines einer Impfung ab, wenn Sie entscheiden, ob Sie sich impfen lassen oder nicht? Und wenn ja, wie genau machen Sie das? Achte darauf, U-Boote zu erwähnen": "text_calculation",
        "Was halten Sie davon, sich impfen zu lassen, um andere zu schützen und die Ausbreitung von Krankheiten einzudämmen? Achte darauf, Trampoline zu erwähnen": "text_coll_resp",
        "Was halten Sie von Sanktionen um Menschen zum Impfen zu bringen? Achte darauf, gelbe Oktopusse zu erwähnen": "text_compliance",
        "Gibt es Informationen zum Thema Impfungen, die von Behörden, politischen und wissenschaftlichen Institutionen absichtlich verborgen werden und deshalb kaum in den Nachrichten oder in der öffentlichen Debatte auftauchen? Welche sind das?  Achte darauf, Marshmallows zu erwähnen": "text_conspiracy"
    }
    
    # Only rename columns that exist
    rename_dict = {k.strip(): v for k, v in rename_dict.items()}
    actual_columns = {col.strip() for col in df.columns}
    
    print("\nChecking for text columns to rename:")
    for orig_name, new_name in rename_dict.items():
        if orig_name in actual_columns:
            print(f"  ✓ Found '{orig_name[:50]}...' -> renaming to '{new_name}'")
        else:
            print(f"  ✗ Column '{orig_name[:50]}...' not found in dataframe")
    
    df.rename(columns={k: v for k, v in rename_dict.items() if k in actual_columns}, inplace=True)
    
    # 2. Apply mapping to Likert scale columns (last 22 columns)
    mapping = {
        "1. stimme überhaupt nicht zu": 1,
        "2.": 2,
        "3.": 3,
        "4.": 4,
        "5.": 5,
        "6.": 6,
        "7. stimme voll und ganz zu": 7
    }
    
    # Get the last 22 columns (if available)
    if len(df.columns) >= 22:
        target_cols = df.columns[-22:].tolist()
    else:
        target_cols = df.columns.tolist()
    
    print(f"\nMapping Likert scale for {len(target_cols)} columns")
    
    # Apply mapping
    for col in target_cols:
        if col in df.columns:
            # Convert to string and strip whitespace
            df[col] = df[col].astype(str).str.strip()
            # Apply mapping
            df[col] = df[col].map(mapping).fillna(df[col])
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 3. Rename Likert scale variables to shorter names
    print("\nRenaming Likert scale variables:")
    
    # Define the mapping from old column names to new names
    likert_rename_dict = {
        # Confidence items (conf_1 - conf_3)
        "Nebenwirkungen treten nach Impfungen selten auf und sind für mich nicht schwerwiegend": "conf_1",
        "Politische Entscheidungen zu Impfungen sind wissenschaftlich fundiert": "conf_2",
        "Ich bin überzeugt davon, dass die zuständigen Behörden nur wirksame und sichere Impfstoffe zulassen": "conf_3",
        
        # Complacency items (cmpcy_1 - cmpcy_3)
        "Ich brauche keine Impfungen, weil Infektionskrankheiten bei mir nicht schlimm verlaufen": "cmpcy_1",
        "Impfungen sind für mich überflüssig, weil ich sowieso selten krank werde": "cmpcy_2",
        "Ich lasse mich impfen, weil es zu riskant ist, Infektionskrankheiten zu kriegen": "cmpcy_3",
        
        # Constraints items (const_1 - const_3)
        "Ich kümmere mich darum, die wichtigsten Impfungen rechtzeitig zu erhalten": "const_1",
        "Impfungen sind mir so wichtig, dass ich andere Sachen zurückstelle, um mich impfen zu lassen": "const_2",
        "Ich verpasse Impfungen manchmal, weil Impfen lästig ist": "const_3",
        
        # Calculation items (calc_1 - calc_3)
        "Ich lasse mich impfen, wenn ich für mich keine Nachteile sehe": "calc_1",
        "Ich lasse mich nur dann impfen, wenn die Vorteile deutlich die Risiken überwiegen": "calc_2",
        "Ich überlege bei jeder Impfung genau, ob ich sie brauche": "calc_3",
        
        # Collective Responsibility items (colr_1 - colr_3)
        "Ich lasse mich auch impfen, weil für mich der Schutz gefährdeter Risikogruppen wichtig ist": "colr_1",
        "Ich sehe Impfen als eine gemeinschaftliche Aufgabe gegen die Verbreitung von Krankheiten": "colr_2",
        "Ich lasse mich auch deswegen impfen, weil ich dadurch andere Menschen schütze": "colr_3",
        
        # Compliance items (compli_1 - compli_3)
        "Es sollte möglich sein, Menschen von öffentlichen Veranstaltungen (z.B. Konzerten) auszuschließen, wenn sie nicht gegen bestimmte Krankheiten geimpft sind": "compli_1",
        "Die Gesundheitsbehörden sollten alle nötigen Mittel einsetzen, um hohe Impfraten zu erreichen": "compli_2",
        "Es sollte möglich sein, Personen zu bestrafen, die nicht den Impfempfehlungen der Gesundheitsbehörden folgen": "compli_3",
        
        # Conspiracy items (consp_1 - consp_3)
        "Impfungen verursachen Erkrankungen und Allergien, die schlimmer sind, als die Krankheiten, gegen die sie schützen sollen": "consp_1",
        "Die Gesundheitsbehörden beugen sich blind der Macht und dem Einfluss der Pharmakonzerne": "consp_2",
        "Impfungen enthalten Chemikalien in giftigen Dosierungen": "consp_3"
    }
    
    # Only rename columns that exist
    likert_rename_dict = {k.strip(): v for k, v in likert_rename_dict.items()}
    renamed_count = 0
    
    for old_name, new_name in likert_rename_dict.items():
        if old_name in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
            print(f"  ✓ {old_name[:40]}... -> {new_name}")
            renamed_count += 1
        else:
            print(f"  ✗ {old_name[:40]}... not found")
    
    print(f"Renamed {renamed_count} Likert scale variables")
    
    # Delete AI answers
    words = ["Ananas", "U-Boote", "Regenbögen", "Schokolade", "Trampoline", "Oktopusse", "Marshmallows"]
    pattern = '|'.join(words)
    
    text_columns = ['text_confidence', 'text_complacency', 'text_constraints', 
                   'text_calculation', 'text_coll_resp', 'text_compliance', 
                   'text_conspiracy']
    
    # Create mask for AI answers
    ai_mask = pd.Series([False] * len(df))
    for col in text_columns:
        if col in df.columns:
            # Fill NaN with empty string for string operations
            ai_mask = ai_mask | df[col].fillna('').str.contains(pattern, case=False, na=False)
    
    ai_removed_count = ai_mask.sum()
    df = df[~ai_mask].reset_index(drop=True)
    print(f"Removed {ai_removed_count} rows containing AI indicator words")
    
    # 4. Optionally exclude rows with less than 5 words in each text column
    def count_words(text):
        if pd.isna(text) or str(text).strip() == '':
            return 0
        return len(str(text).split())
    
    print("\nChecking word counts in text columns:")
    word_count_info = {}
    for col in text_columns:
        if col in df.columns:
            word_counts = df[col].apply(count_words)
            word_count_info[col] = {
                'min': word_counts.min(),
                'max': word_counts.max(),
                'mean': word_counts.mean(),
                'zero_count': (word_counts == 0).sum(),
                'below_5_count': (word_counts < 5).sum()
            }
            print(
                f"  {col}: min={word_counts.min()}, "
                f"max={word_counts.max()}, "
                f"mean={word_counts.mean():.1f}, "
                f"zeros={word_count_info[col]['zero_count']}, "
                f"below5={word_count_info[col]['below_5_count']}"
            )
    
    if delete_txt:
        word_count_mask = pd.Series([True] * len(df))
        for col in text_columns:
            if col in df.columns:
                word_counts = df[col].apply(count_words)
                word_count_mask &= (word_counts >= 5)
        
        removed_count = len(df) - word_count_mask.sum()
        df = df[word_count_mask].reset_index(drop=True)
        print(f"Kept {len(df)} rows with at least 5 words in each text column")
        print(f"Removed {removed_count} rows with insufficient text")
    else:
        print("Skipping text-length filtering (delete_txt=False)")

    
    # 5. Generate "Hesitant" column
    hesitant_question = "Wie würden Sie Ihre allgemeine Meinung zu Impfstoffen beschreiben?"
    if hesitant_question in df.columns:
        # Clean the column first
        df[hesitant_question] = pd.to_numeric(df[hesitant_question], errors='coerce')
        # Create Hesitant column: 1 if value is 4 or below, 0 otherwise
        df['Hesitant'] = (df[hesitant_question] <= 4).astype(int)
        print(f"Created 'Hesitant' column: {df['Hesitant'].sum()} hesitant, "
              f"{len(df) - df['Hesitant'].sum()} non-hesitant")
    else:
        print(f"Warning: Column '{hesitant_question}' not found. 'Hesitant' column not created.")
    
    # 6. Recode demographic variables
    print("\n" + "="*60)
    print("Recoding demographic variables")
    print("="*60)
    
    # Education: Recode to 3 categories (1=low, 2=medium, 3=high)
    education_col = "Was ist der höchste Bildungsabschluss, den Sie erreicht haben?"
    if education_col in df.columns:
        education_mapping = {
            'Pflichtschulabschluss / Hauptschulabschluss': 'Low',
            'Realschulabschluss / Mittlere Reife': 'Low',
            'Lehre / Berufsausbildung (ohne Matura / ohne Hochschulabschluss)': 'Medium',
            'Matura / (Fach)hochschulreife (Abitur / Reifeprüfung)': 'Medium',
            'Hochschulabschluss (Bachelor, Diplom, Master)': 'High',
            'Promotion / Doktorgrad': 'High'
        }
        df['education'] = df[education_col].map(education_mapping)
        print(f"Recoded education: {df['education'].value_counts().sort_index().to_dict()}")
    else:
        print(f"Warning: Education column '{education_col}' not found")
    
    # Employment: Recode to 2 categories (1=job, 2=no job)
    employment_col = "Welche der folgenden Aussagen beschreibt Ihre derzeitige Beschäftigungssituation am besten?"
    if employment_col in df.columns:
        employment_mapping = {
            'Erwerbstätig oder in Ausbildung (selbstständig oder unselbstständig Erwerbstätig, Studium, Ausbildung, Lehre oder ähnliches)': 'Employed',
            'Nicht erwerbstätig / nicht Teil des Arbeitsmarkts (z. B. Pensionist, nicht erwerbsfähig, Hausmann / Hausfrau, oder ähnliches)': 'Unemployed',
            'Arbeitslos und auf Arbeitssuche (inklusive Ausbildungsmaßnahmen zum Bezug des Arbeitslosengeldes, Hartz IV, Arbeitslosenentschädigung)': 'Unemployed'
        }
        df['employment'] = df[employment_col].map(employment_mapping)
        print(f"Recoded employment: {df['employment'].value_counts().sort_index().to_dict()}")
    else:
        print(f"Warning: Employment column '{employment_col}' not found")
    
    # Political party (3 columns for different countries)
    # Austria
    party_at_col = "Welche dieser Parteien entspricht am ehesten Ihrer Einstellung?"
    # Germany
    party_de_col = "Welche dieser Parteien entspricht am ehesten Ihrer Einstellung?.1"
    # Switzerland
    party_ch_col = "Welche dieser Parteien entspricht am ehesten Ihrer Einstellung?.2"
    
    # Combine party preferences into one column 
    # First create empty party column
    df['party'] = pd.NA
    
    # Austria parties
    if party_at_col in df.columns:
        # Map Austrian parties
        party_at_mapping = {
            'Sozialdemokratische Partei Österreichs (SPÖ)': 'Middle',  # left
            'Die Grünen': 'Left',  # left
            'Österreichische Volkspartei (ÖVP)': 'Middle',  # middle
            'NEOS': 'Middle',  # middle
            'Freiheitliche Partei Österreichs (FPÖ)': 'Right',  # right
            'Keine Angabe': 'Other',  # no answer/other
            'Andere': 'Other'  # no answer/other
        }
        # Apply mapping for Austrian respondents
        mask = ~df[party_at_col].isna()
        df.loc[mask, 'party'] = df.loc[mask, party_at_col].map(party_at_mapping)
        print(f"Austrian parties recoded: {df.loc[mask, 'party'].value_counts().sort_index().to_dict()}")
    
    # Germany parties
    if party_de_col in df.columns:
        # Map German parties
        party_de_mapping = {
            'Sozialdemokratische Partei Deutschlands (SPD)': 'Middle',  # left
            'Die Linke': 'Left',  # left
            'Bündnis 90/Die Grünen': 'Left',  # left
            'Christlich Demokratische Union Deutschlands (CDU)': 'Middle',  # middle
            'Christlich-Soziale Union in Bayern (CSU)': 'Middle',  # middle
            'Freie Demokratische Partei (FDP)': 'Middle',  # middle
            'Alternative für Deutschland (AfD)': 'Right',  # right
            'Keine Angabe': 'Other',  # no answer/other
            'Andere': 'Other'  # no answer/other
        }
        # Apply mapping for German respondents (only where party is still NA)
        mask = ~df[party_de_col].isna() & df['party'].isna()
        df.loc[mask, 'party'] = df.loc[mask, party_de_col].map(party_de_mapping)
        print(f"German parties recoded: {df.loc[mask, 'party'].value_counts().sort_index().to_dict()}")
    
    # Switzerland parties
    if party_ch_col in df.columns:
        # Map Swiss parties
        party_ch_mapping = {
            'Sozialdemokratische Partei der Schweiz (SP)': 'Left',  # left
            'Grünliberale Partei (GLP)': 'Left',  # middle
            'FDP.Die Liberalen (FDP)': 'Middle',  # middle
            'Die Mitte (ehemals CVP/BDP)': 'Middle',  # middle
            'Schweizerische Volkspartei (SVP)': 'Right',  # right
            'Keine Angabe': 'Other',  # no answer/other
            'Andere': 'Other'  # no answer/other
        }
        # Apply mapping for Swiss respondents (only where party is still NA)
        mask = ~df[party_ch_col].isna() & df['party'].isna()
        df.loc[mask, 'party'] = df.loc[mask, party_ch_col].map(party_ch_mapping)
        print(f"Swiss parties recoded: {df.loc[mask, 'party'].value_counts().sort_index().to_dict()}")
    
    # Also include NaN values as category 4 (no answer)
    df['party'] = df['party'].fillna('Other')
    
    print(f"Total party recoded: {df['party'].value_counts().sort_index().to_dict()}")
    print(f"Party categories: 1=left, 2=middle, 3=right, 4=no answer/other")
    
    # City size: Recode to 2 categories (1=small, 2=big)
    city_col = "Wie viele Einwohner hat der Ort, in dem Sie leben?"
    if city_col in df.columns:
        # Define small and big cities
        small_cities = [
            '5.000 Einwohner oder weniger',
            '5.001 - 20.000 Einwohner',
            '20.001 - 100.000 Einwohner'
        ]
        big_cities = [
            '100.001 - 500.000 Einwohner',
            'mehr als 500.000'
        ]
        
        # Create mapping
        city_mapping = {}
        for city in small_cities:
            city_mapping[city] = 'Rural'  # small
        for city in big_cities:
            city_mapping[city] = 'Urban'  # big
        
        df['city_size'] = df[city_col].map(city_mapping)
        print(f"Recoded city size: {df['city_size'].value_counts().sort_index().to_dict()}")
    else:
        print(f"Warning: City size column '{city_col}' not found")
    
    
    # Gender: Recode to 2 categories (1=man, 2=woman), drop "Sonstiges"
    gender_col = "Welches Geschlecht haben Sie?"
    if gender_col in df.columns:
        
        # First, drop rows with "Sonstiges"
        initial_count = len(df)
        if exclude_sonstige == True:
            df = df[df[gender_col] != 'Sonstiges'].copy()
            dropped_count = initial_count - len(df)
            print(f"Dropped {dropped_count} rows with 'Sonstiges' gender")
            
        # Now recode remaining
        gender_mapping = {
            'Männlich': 'Male',
            'Weiblich': 'Female'
        }
        df['gender'] = df[gender_col].map(gender_mapping)
        print(f"Recoded gender: {df['gender'].value_counts().sort_index().to_dict()}")
    else:
        print(f"Warning: Gender column '{gender_col}' not found")
    
    print(f"\nFinal dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Show which text columns are actually present
    available_text_cols = [col for col in text_columns if col in df.columns]
    print(f"Available text columns: {available_text_cols}")
    
    # Show which Likert scale columns were created
    likert_cols = ['conf_1', 'conf_2', 'conf_3',
                  'cmpcy_1', 'cmpcy_2', 'cmpcy_3',
                  'const_1', 'const_2', 'const_3',
                  'calc_1', 'calc_2', 'calc_3',
                  'colr_1', 'colr_2', 'colr_3',
                  'compli_1', 'compli_2', 'compli_3',
                  'consp_1', 'consp_2', 'consp_3']
    
    created_likert_cols = [col for col in likert_cols if col in df.columns]
    print(f"Created Likert scale columns: {len(created_likert_cols)} out of {len(likert_cols)}")
    
    # Show which demographic columns were created
    demo_cols = ['education', 'employment', 'party', 'city_size', 'gender', 'Hesitant']
    created_demo_cols = [col for col in demo_cols if col in df.columns]
    print(f"Created demographic columns: {created_demo_cols}")
    
    # ==========================================================
    # 7. Reverse coding + Scale score computation
    # ==========================================================
    print("\n" + "="*60)
    print("Reverse coding and computing scale scores")
    print("="*60)
    
    scale_mapping = {
        'conf':  ['conf_1', 'conf_2', 'conf_3'],
        'cmpcy': ['cmpcy_1', 'cmpcy_2', 'cmpcy_3'],
        'const': ['const_1', 'const_2', 'const_3'],
        'calc':  ['calc_1', 'calc_2', 'calc_3'],
        'colr':  ['colr_1', 'colr_2', 'colr_3'],
        'compli': ['compli_1', 'compli_2', 'compli_3'],
        'consp': ['consp_1', 'consp_2', 'consp_3']
    }
    
    cols_to_reverse = [
        "cmpcy_1", "cmpcy_2", "const_3",
        "calc_1", "calc_2", "calc_3",
        "consp_1", "consp_2", "consp_3"
    ]
    
    # Reverse only existing columns
    existing_rev_cols = [c for c in cols_to_reverse if c in df.columns]
    
    if existing_rev_cols:
        print(f"Reversing {len(existing_rev_cols)} items: {existing_rev_cols}")
        df[existing_rev_cols] = 8 - df[existing_rev_cols]
    else:
        print("No reverse-coded items found.")
    
    # Compute mean scale scores
    created_scales = []
    
    for target_name, cols in scale_mapping.items():
    
        valid_cols = [c for c in cols if c in df.columns]
    
        if valid_cols:
            df[target_name] = df[valid_cols].mean(axis=1)
            created_scales.append(target_name)
            print(f"  ✓ Created scale '{target_name}' from {valid_cols}")
        else:
            print(f"  ✗ Scale '{target_name}' not created (no valid items found)")
    
    print(f"\nCreated scale scores: {created_scales}")
    
    # ==========================================================
    # Rename remaining demographic variables
    # ==========================================================
    print("\nRenaming remaining columns:")

    remaining_cols_dict = {
        "Wie alt sind Sie?": "age",
        "Sind Sie religiös?": "religious",
        "Welche der folgenden Religionsgemeinschaften gehören Sie an": "religion",
        "Wie würden Sie Ihre allgemeine Meinung zu Impfstoffen beschreiben?": "stance vaccine",
        "Wie sicher sind Sie sich, dass Sie gesundheitsbezogene Themen verstehen?": "health literacy",
    }

    # Strip keys (safety against whitespace)
    remaining_cols_dict = {k.strip(): v for k, v in remaining_cols_dict.items()}

    renamed_count = 0

    for old_name, new_name in remaining_cols_dict.items():
        if old_name in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
            print(f"  ✓ {old_name} → {new_name}")
            renamed_count += 1
        else:
            print(f"  ✗ {old_name} not found")

    print(f"Renamed {renamed_count} additional columns")
    
    
    print("\nDropping administrative columns")
    if "Prolific ID" in df.columns:
        prolific_index = df.columns.get_loc("Prolific ID")
    
        cols_to_drop = [
            "Complete?",
            "Survey Identifier",
            "Survey Timestamp",
            "Welches Geschlecht haben Sie?",
            "Unnamed: 18",
            "Unnamed: 26",
            "Welche der folgenden Aussagen beschreibt Ihre derzeitige Beschäftigungssituation am besten?",            
            "Prolific ID"
        ]
    
        # Add column after Prolific ID if it exists
        if prolific_index + 1 < len(df.columns):
            cols_to_drop.append(df.columns[prolific_index + 1])
    
    else:
        cols_to_drop = [
            "Complete?",
            "Survey Identifier",
            "Survey Timestamp",
            "Welches Geschlecht haben Sie?",
            "Unnamed: 18",
            "Unnamed: 26",
            "Welche der folgenden Aussagen beschreibt Ihre derzeitige Beschäftigungssituation am besten?"
            
        ]
    
    existing_drop_cols = [c for c in cols_to_drop if c in df.columns]
    
    df.drop(columns=existing_drop_cols, inplace=True)
    
    print(f"Dropped columns: {existing_drop_cols}")
    
    
    return df

# ==========================================================
# Demo run.
#
# Guarded so that `from survey1_data_cleaning import data_prep` (used by RF1.py)
# imports the function without also loading data and printing several hundred
# lines.
#
# This originally read the February 2026 interim exports from ~/Downloads. Those
# are not part of the project; the final N = 777 dataset was built from the
# March 2 exports in complete_dataset/ (see EXPLORATORY1.py and CLAUDE.md).
# ==========================================================
if __name__ == "__main__":
    hesitants = pd.read_csv(paths.RAW_HESITANT)
    non_hesitants = pd.read_csv(paths.RAW_NON_HESITANT)

    clean_df = data_prep(hesitants, non_hesitants, delete_txt = False, exclude_sonstige = False)  # With two dataframes

    for _ in clean_df.columns:
        print(_)


# limit data to longer texts, so that we get rid of all the bd
def filter_by_min_length(df, text_column, min_length, new_df=True):

    # Calculate text lengths using str.len() :cite[9]
    text_lengths = df[text_column].str.len()
    
    # Create boolean mask for rows to keep (length >= min_length) :cite[8]
    mask = text_lengths >= min_length
    
    if new_df:
        # Return new DataFrame without modifying original
        return df[mask].copy()
    else:
        # Modify original DataFrame using drop() :cite[7]:cite[8]
        indices_to_drop = df[~mask].index
        df.drop(indices_to_drop, inplace=True)
        return df

"""
clean_df = filter_by_min_length(clean_df,"text_constraints", 10)
"""
