def split_cabin(x):
    if len(str(x).split('/')) < 3:
        return ['Missing', '-1', 'Missing']
    else:
        return str(x).split('/')


# Create a preprocessing function to transform our dataset
def preprocessing(df):
    # Fill missing values in homeplanet with missing
    df['HomePlanet'] = df['HomePlanet'].fillna('Missing')
    # Cryosleep - highly correlated - drop na rows
    df['CryoSleep'] = df['CryoSleep'].fillna('Missing')

    # Cabin preprocessing - extract Deck and Side
    df['TempCabin'] = df['Cabin'].apply(lambda x: split_cabin(x))
    df['Deck'] = df['TempCabin'].apply(lambda x: x[0])
    df['Cabin_Number'] = df['TempCabin'].apply(lambda x: int(x[1]))
    df['Cabin_Type'] = df['TempCabin'].apply(lambda x: x[2])
    df.drop(['TempCabin'], axis=1, inplace=True)
    df.drop(['Cabin'], axis=1, inplace=True)
    # Destination
    df['Destination'] = df['Destination'].fillna('Missing')
    # Age use median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    # VIP - drop na rows
    df['VIP'] = df['VIP'].fillna('Missing')

    # Monetary spending columns
    df['RoomService'] = df['RoomService'].fillna(df['RoomService'].mean())
    df['FoodCourt'] = df['FoodCourt'].fillna(df['FoodCourt'].mean())
    df['ShoppingMall'] = df['ShoppingMall'].fillna(df['ShoppingMall'].mean())
    df['Spa'] = df['Spa'].fillna(df['Spa'].mean())
    df['VRDeck'] = df['VRDeck'].fillna(df['VRDeck'].mean())

    # Drop name due to high cardinality
    df.drop('Name', axis=1, inplace=True)
