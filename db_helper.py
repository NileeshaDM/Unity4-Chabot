from pymongo import MongoClient

# MongoDB connection string and database name
connection_string = 'mongodb+srv://Unityfour:0767@cluster0.sghz5ld.mongodb.net/?retryWrites=true&w=majority'
database_name = 'Unity4db'

try:
    client = MongoClient(connection_string)
    DB = client[database_name]
    collection = DB['food_Items']
    collection2=DB['order_tracking']
    collection3=DB['orders']
    print("Connected to MongoDB successfully.")

except Exception as e:
    print(f"Error connecting to MongoDB: {str(e)}")
