import pandas as pd

df = pd.read_csv("fraud_results_artists.csv")

HIP_HOP = ["Drake", "Kanye West", "Lil Wayne", "Lil Baby", "Lil Uzi Vert",
           "Lil Durk", "Lil Nas X", "Lil Tecca", "Lil Tjay", "Lil Mosey",
           "Lil Skies", "Lil Pump", "Lil Peep", "Lil Xan", "Lil Yachty",
           "21 Savage", "Travis Scott", "Post Malone", "Roddy Ricch",
           "DaBaby", "Juice WRLD", "XXXTENTACION", "Future", "Young Thug",
           "Gunna", "Migos", "Cardi B", "Nicki Minaj", "Megan Thee Stallion",
           "J. Cole", "Kendrick Lamar", "Eminem", "Jay-Z", "JAY-Z",
           "Playboi Carti", "Trippie Redd", "6ix9ine", "Polo G",
           "Pop Smoke", "NBA YoungBoy", "YoungBoy Never Broke Again",
           "King Von", "Kodak Black", "Kevin Gates", "Meek Mill", "Wiz Khalifa",
           "Big Sean", "Chance the Rapper", "Kid Cudi", "Mac Miller",
           "Logic", "G-Eazy", "Macklemore", "Tyga", "YG", "Gucci Mane",
           "2 Chainz", "Quavo", "Offset", "Metro Boomin", "Mustard",
           "DJ Khaled", "French Montana", "A$AP Rocky", "A$AP Ferg",
           "Jack Harlow", "The Kid LAROI", "Moneybagg Yo",
           "NLE Choppa", "Sleepy Hallow", "Tay-K", "Blueface",
           "BlocBoy JB", "Comethazine", "Smokepurpp", "Ski Mask The Slump God",
           "YNW Melly", "Tee Grizzley", "G Herbo", "Joyner Lucas",
           "Internet Money", "JACKBOYS", "Quality Control", "Dreamville",
           "Young Stoner Life", "NAV", "Swae Lee", "Rae Sremmurd",
           "Fetty Wap", "Tory Lanez", "WizKid", "Wale", "Nipsey Hussle",
           "ScHoolboy Q", "Jay Rock", "Childish Gambino", "Anderson .Paak",
           "BROCKHAMPTON", "Desiigner", "Sheck Wes", "CJ", "Calboy",
           "Stunna 4 Vegas", "$uicideboy$", "NF", "Machine Gun Kelly",
           "24kGoldn", "iann dior", "Joji", "blackbear", "Powfu",
           "Trevor Daniel", "Surfaces", "PnB Rock"]

RNB = ["SZA", "H.E.R.", "Jhené Aiko", "Summer Walker", "Ella Mai",
       "Kehlani", "Normani", "Brent Faiyaz", "Daniel Caesar", "Frank Ocean",
       "Miguel", "Ty Dolla $ign", "6LACK", "PARTYNEXTDOOR", "Don Toliver",
       "Rod Wave", "Giveon", "Lucky Daye", "Bryson Tiller", "Trey Songz",
       "Alicia Keys", "John Mayer", "Bruno Mars", "Khalid", "Alessia Cara",
       "Madison Beer", "Zendaya", "WILLOW", "Chris Brown", "Usher",
       "Ne-Yo", "Mario", "Tank", "Jacquees", "PnB Rock",
       "Jeremih", "Ty Dolla $ign", "Tinashe", "Kiana Ledé",
       "Lucky Daye", "Victoria Monét", "Ari Lennox", "Masego",
       "Joyce Wrice", "Snoh Aalegra", "Ravyn Lenae", "Ann Marie"]

POP = ["Taylor Swift", "Ed Sheeran", "Ariana Grande", "Billie Eilish",
       "Justin Bieber", "Selena Gomez", "Shawn Mendes", "Harry Styles",
       "Dua Lipa", "Olivia Rodrigo", "Demi Lovato", "Miley Cyrus",
       "Katy Perry", "Lady Gaga", "Maroon 5", "Coldplay", "OneRepublic",
       "Imagine Dragons", "Twenty One Pilots", "Panic! At The Disco",
       "The Chainsmokers", "Jonas Brothers", "Little Mix", "BTS", "BLACKPINK",
       "Halsey", "Lorde", "Lana Del Rey", "Camila Cabello", "Charlie Puth",
       "Bebe Rexha", "Ava Max", "Anne-Marie", "Jason Derulo", "Clean Bandit",
       "Sam Smith", "Lewis Capaldi", "James Arthur", "Niall Horan",
       "Liam Payne", "ZAYN", "Hailee Steinfeld", "Julia Michaels",
       "Lizzo", "Kesha", "Sia", "P!nk", "Conan Gray", "Tate McRae",
       "Sabrina Carpenter", "Troye Sivan", "Zara Larsson", "Ellie Goulding",
       "Rita Ora", "The Weeknd", "Adele", "Elton John", "Justin Timberlake",
       "John Legend", "Rihanna", "Beyoncé", "GAYLE", "Glass Animals",
       "Tones And I", "Dean Lewis", "JP Saxe", "Jeremy Zucker", "Lauv",
       "Alec Benjamin", "Ali Gatie", "Bazzi", "bbno$", "beabadoobee",
       "Wallows", "Clairo", "Melanie Martinez", "Ashnikko", "King Princess",
       "AURORA", "AJR", "Grouplove", "Lord Huron", "Portugal. The Man",
       "Hozier", "Arctic Monkeys", "The Killers", "The Neighbourhood",
       "Paramore", "Fleetwood Mac", "Journey", "The Weeknd"]

LATIN = ["Bad Bunny", "J Balvin", "Ozuna", "Maluma", "Daddy Yankee",
         "Anuel AA", "KAROL G", "Shakira", "Luis Fonsi", "Nicky Jam",
         "Farruko", "Sech", "Myke Towers", "Jhay Cortez", "Tainy",
         "Becky G", "ROSALÍA", "Aventura", "Prince Royce", "Pedro Capó",
         "Casper Magico", "Darell", "DJ Luian", "Mambo Kingz", "Nio Garcia",
         "Eslabon Armado", "Wisin", "Natti Natasha", "Rauw Alejandro"]

COUNTRY = ["Morgan Wallen", "Luke Combs", "Luke Bryan", "Blake Shelton",
           "Jason Aldean", "Carrie Underwood", "Thomas Rhett", "Kane Brown",
           "Kacey Musgraves", "Maren Morris", "Chris Stapleton",
           "Florida Georgia Line", "Dan + Shay", "Old Dominion", "Brett Young",
           "Jordan Davis", "Russell Dickerson", "Dustin Lynch", "Billy Ray Cyrus",
           "Chris Young", "Lee Brice", "Brett Eldredge", "Jon Pardi",
           "Billy Currington", "Kelsea Ballerini", "Gabby Barrett",
           "Walker Hayes", "LANCO", "Sam Hunt", "Zac Brown Band"]

EDM = ["Calvin Harris", "David Guetta", "Martin Garrix", "Avicii",
       "Marshmello", "Diplo", "Major Lazer", "Skrillex", "Steve Aoki",
       "Tiësto", "Kygo", "ILLENIUM", "Loud Luxury", "Joel Corry",
       "MEDUZA", "Regard", "Jonas Blue", "Cheat Codes", "Mark Ronson",
       "DJ Snake", "Disclosure", "Gesaffelstein", "Cashmere Cat"]

def assign_genre(artist):
    if artist in LATIN:
        return "Latin"
    elif artist in COUNTRY:
        return "Country"
    elif artist in EDM:
        return "EDM / Electronic"
    elif artist in RNB:
        return "R&B / Soul"
    elif artist in HIP_HOP:
        return "Hip Hop / Rap"
    elif artist in POP:
        return "Pop"
    else:
        return "Other"

df["genre_category"] = df["artist"].apply(assign_genre)
df["genres"] = df["genre_category"]
df.to_csv("artist_genres.csv", index=False)

print("✅ Genres assigned:")
print(df["genre_category"].value_counts().to_string())
print(f"\nSample R&B artists:")
print(df[df["genre_category"] == "R&B / Soul"]["artist"].tolist())
