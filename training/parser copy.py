import shapefile
import csv
import os
import glob

def process_shapefile():
    # 1. Automatically find the shapefile in the current directory
    shp_files = glob.glob("*.shp")
    
    if not shp_files:
        print("Error: No .shp file found in the current directory.")
        return
        
    shp_path = shp_files[0]
    if len(shp_files) > 1:
        print(f"Multiple shapefiles found. Using the first one: {shp_path}")
        
    # 2. Generate output file names based on the found shapefile
    base_name = os.path.splitext(shp_path)[0]
    prefix = base_name[:2]
    test_csv_path = f"{prefix}_test_generated.csv"
    train_csv_path = f"{prefix}_training_generated.csv"
    
    try:
        sf = shapefile.Reader(shp_path)
    except Exception as e:
        print(f"Failed to open shapefile: {e}")
        return
        
    fields = [field[0] for field in sf.fields[1:]]

    #Texas why.
    start_idx = 7 if prefix == 'tx' else 5
    
    # 3. Field Configuration
    GEOID_FIELD = 'GEOID20'
    LON_FIELD = 'INTPTLON20'    
    LAT_FIELD = 'INTPTLAT20'    
    
    #Population
    VALUE1_FIELD = 'VAP_MOD'
    VALUE2_FIELD = 'G20PRERTRU'
    VALUE3_FIELD = 'G20PREDBID'

    #The sixth (party) letter is after "G20[3-char position]"
    r_idxs = [i for i, item in enumerate(fields[start_idx:], start=start_idx) if item[6] == 'R']
    d_idxs = [i for i, item in enumerate(fields[start_idx:], start=start_idx) if item[6] == 'D']
    #There are too many libertarian, independent, other and unaffiliated candidates. Get all presidential candidates and filter out R and D.
    i_idxs = [i for i, item in enumerate(fields[start_idx:], start=start_idx) if i not in r_idxs and i not in d_idxs]

    required_fields = [GEOID_FIELD, VALUE1_FIELD, VALUE2_FIELD, VALUE3_FIELD]
    missing_fields = [f for f in required_fields if f not in fields]
    
    if missing_fields:
        print(f"Error: Missing fields {missing_fields}.")
        print(f"Available fields are: {fields}")
        return

    geoid_idx = fields.index(GEOID_FIELD)
    pop_idx = fields.index(VALUE1_FIELD)


    
    has_internal_points = (LON_FIELD in fields) and (LAT_FIELD in fields)
    if not has_internal_points:
        print("Internal point attributes not found. Calculating centroids from bounding boxes...")

    print(f"Processing '{shp_path}'...")

    # 4. Open BOTH CSV files simultaneously to save processing time
    with open(test_csv_path, 'w', newline='') as test_file, \
         open(train_csv_path, 'w', newline='') as train_file:
        
        test_writer = csv.writer(test_file)
        train_writer = csv.writer(train_file)
        
        for shape_rec in sf.shapeRecords():
            rec = shape_rec.record

            geoid = rec[geoid_idx]
            
            try:
                pop = int(float(rec[pop_idx])) 
            except (ValueError, TypeError):
                pop = 0

            try:
                r = sum(float(rec[i]) for i in r_idxs)
            except (ValueError, TypeError):
                senr = 0.0

            try:
                d = sum(float(rec[i]) for i in d_idxs)
            except (ValueError, TypeError):
                send = 0.0
                
            try:
                i = sum(float(rec[i]) for i in i_idxs)
            except (ValueError, TypeError):
                seni = 0.0            
            
            if has_internal_points:
                lon = float(rec[lon_idx])
                lat = float(rec[lat_idx])
            else:
                bbox = shape_rec.shape.bbox
                lon = (bbox[0] + bbox[2]) / 2.0
                lat = (bbox[1] + bbox[3]) / 2.0

            #Total votes
            votes = r + d + i

            #dont divide by 0
            r_ratio = 0 if votes == 0 else r / votes
            d_ratio = 0 if votes == 0 else d / votes

            votes = round(votes, 2)
            
            # Write 4-column format to Test CSV
            test_writer.writerow([geoid, lon, lat, pop])
            
            # Write 7-column format to Training CSV
            train_writer.writerow([geoid, lon, lat, pop, votes, r_ratio, d_ratio])

    print(f"Successfully generated: {test_csv_path}")
    print(f"Successfully generated: {train_csv_path}")

if __name__ == "__main__":
    process_shapefile()