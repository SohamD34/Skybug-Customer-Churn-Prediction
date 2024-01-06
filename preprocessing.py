def country_mapping(data):
    country_code = {'France':0, 'Spain':1, 'Germany':2}
    data['Geography'] = data['Geography'].map(country_code)
    return data

def gender_mapping(data):
    gender_map = {'Male':0, 'Female':1}
    data['Gender'] = data['Gender'].map(gender_map)
    return data

def scaling(data, columns, scaler):
    max_vals = []
    min_vals = []
    for col in columns:
        max_vals.append(data[col].max())
        min_vals.append(data[col].min())
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
    return max_vals, min_vals, columns, data
