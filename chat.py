import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import json
import random
import db_helper as db

#nltk.download('stopwords')

# Load the model and data files
model = load_model('foodbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def preprocess_sentence(sentence):
    # sw = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    sentence_words = nltk.word_tokenize(sentence)
    # sentence_words = [i for i in sentence_words if i not in sw]
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence, words, show_details=True):
    sentence_words = preprocess_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def class_prediction(sentence, model):
    p = bag_of_words(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(intents, intents_json):
    tag = intents[0]['intent']
    list_of_intents = intents_json

    for intent in list_of_intents:
        if intent['tag'] == tag:
            responses = intent['responses']
            result = random.choice(responses)
    return result


def chatbot_response(msg):
    intents = class_prediction(msg, model)
    intents_json = json.loads(open('intents.json').read())
    response = None

    for intent in intents:

        # food items options =================================================================================================
        if intent['intent'] == 'options':
            # Retrieve all available food items from MongoDB
            food_items = db.collection.find({}, {"name": 1})
            food_list = [food["name"] for food in food_items]
            response = "Our food items are: " + ", ".join(food_list)

        #food items availability =================================================================================================
        if intent['intent'] == 'available.specific':
            # Check if the user is asking about availability for a specific food item
            for food in db.collection.find({"availability": True}, {"name": 1, "availability": 1}):
                if food["name"].lower() in msg.lower():
                    if "availability" in food:
                        response = f"Yes, {food['name']} is available."
                    else:
                        # If there are no promotions, provide a default response
                        response = getResponse(
                            intents, intents_json['intents'])

        # food item prices =======================================================================================
        if intent['intent'] == 'prices':
            print("Detected 'prices' intent")
            user_message = msg.lower()
            food_items = db.collection.find({}, {"name": 1})
            food_list = [food["name"] for food in food_items]

            # Try to find a food name in the user's message
            found_food_name = None
            for food_item in food_list:
                if food_item.lower() in user_message:
                    found_food_name = food_item
                    break

            if found_food_name:
                # Retrieve the price of the found food item from MongoDB
                food_item = db.collection.find_one({"name": found_food_name})
                if food_item:
                    price = food_item["price"]
                    availability = food_item["availability"]
                    if availability == True:
                        response = f"The price of {found_food_name} is Rs.{price}0/=. \n And this item is currently available."
                    else:
                        response = f"The price of {found_food_name} is Rs.{price}0/=. \n But this item is currently not available."

                else:
                    response = getResponse(
                            intents, intents_json['intents'])
            

        # promotions =================================================================================================
        if intent['intent'] == 'promotions.all':
            # All available promotions
            promotion_list = []
            for food in db.collection.find({"availability": True}):
                if "promotions" in food:
                    promotion_list.append(
                        f"{food['name']}: {food['promotions']}")
            if promotion_list:
                response = "Here are the promotions:\n" + \
                    ",".join(promotion_list)
            else:
                response = getResponse(intents, intents_json['intents'])

        if intent['intent'] == 'promotions.specific':
            # Check if the user is asking about promotions for a specific food item
            for food in db.collection.find({"availability": True}, {"name": 1, "promotions": 1}):
                if food["name"].lower() in msg.lower():
                    if "promotions" in food:
                        response = f"Promotion for {food['name']}: {food['promotions']}"
                    else:
                        # If there are no promotions, provide a default response
                        response = getResponse(intents, intents_json['intents'])

        # order tracking =================================================================================================
        if intent['intent'] == 'order.tracking':
            print("Detected 'order.tracking' intent")
            user_message = msg.lower()  # Convert the user's message to lowercase for case-insensitive matching
            print(user_message)

            order_tracking = db.collection2.find({}, {"oId": 1})
            order_ID_list = [id["oId"] for id in order_tracking]

            print(order_tracking)
            print("order id list",order_ID_list)
            
            found_order_ID=None

            for order_id in order_ID_list:
                if order_id.lower() in user_message:
                    found_order_ID = order_id
                    break  # Exit the loop as soon as a match is found

            print(found_order_ID)

            if found_order_ID:
                # Retrieve the price of the found food item from MongoDB
                order_id = db.collection2.find_one({"oId": found_order_ID})
                if order_id:
                    status = order_id["order_status"]
                    print(status)
                    response = f"The order status of {found_order_ID} is {status}"
                    print(response)
                    
                else:
                    response = getResponse(intents, intents_json['intents'])

        # food description =================================================================================================
        if intent['intent'] == 'food.desc':
            print("Detected 'food.desc' intent")
            user_message = msg.lower()  # Convert the user's message to lowercase for case-insensitive matching
            print(user_message)

            food_name = db.collection.find({}, {"name": 1})
            food_name_list = [id["name"] for id in food_name]

            print(food_name)
            print("food item list",food_name_list)
            
            found_food_name=None

            for fname in food_name_list:
                if fname.lower() in user_message:
                    found_food_name = fname
                    break  # Exit the loop as soon as a match is found

            print(found_food_name)

            if found_food_name:
                # Retrieve the name of the found food item from MongoDB
                fname = db.collection.find_one({"name": found_food_name})
                if fname:
                    desc = fname["desc"]
                    print(desc)
                    response = f" {desc}"
                    print(response)
                    
                else:
                    response = getResponse(intents, intents_json['intents'])

        # food type - vegan =================================================================================================
        def get_vegan_food_recommendations():
            # Query MongoDB to retrieve vegan food items
            vegan_foods_cursor = db.collection.find({"type": "vegan"})

            # Extract vegan food names from the cursor
            vegan_food_items = [food_item['name'] for food_item in vegan_foods_cursor]

            return vegan_food_items

        if intent['intent'] == 'food.type':
            print("Detected 'food.desc' intent")
            # Convert the user's message to lowercase for case-insensitive matching
            user_message = msg.lower()
            print(user_message)

            if "vegan" or "vegetarian" in user_message:
                vegan_recommendations = get_vegan_food_recommendations()
                if vegan_recommendations:
                    # Initialize an empty list to store formatted food items
                    foods_list = []
                    for food_item in vegan_recommendations:
                        # Format the food item with a bullet point and add it to the list
                        formatted_food = "- " + food_item
                        foods_list.append(formatted_food)

                    # Join the formatted food items into a single string
                    foods = "\n".join(foods_list)

                    # Construct the response message
                    response = f"Here are some vegan food recommendations:\n{foods}"
                else:
                    response = getResponse(intents, intents_json['intents'])
                    

        # food type - non vegan =================================================================================================
        def get_non_vegan_food_recommendations():
            # Query MongoDB to retrieve vegan food items
            non_vegan_foods_cursor = db.collection.find({"type": "non-vegan"})

            # Extract vegan food names from the cursor
            non_vegan_food_items = [food_item['name'] for food_item in non_vegan_foods_cursor]

            return non_vegan_food_items

        if intent['intent'] == 'food.non_veg_type':
            print("Detected 'food.non_veg_type' intent")
            # Convert the user's message to lowercase for case-insensitive matching
            user_message = msg.lower()
            print(user_message)

            if "non vegan" or "non vegetarian" or "non-vegetarian" or "non-vegan" in user_message:
                non_vegan_recommendations = get_non_vegan_food_recommendations()
                if non_vegan_recommendations:
                    # Initialize an empty list to store formatted food items
                    foods_list = []
                    for food_item in non_vegan_recommendations:
                        # Format the food item with a bullet point and add it to the list
                        formatted_food = "- " + food_item
                        foods_list.append(formatted_food)

                    # Join the formatted food items into a single string
                    foods = "\n".join(foods_list)

                    # Construct the response message
                    response = f"Here are some non-vegan food recommendations:\n{foods}"
                else:
                   response = getResponse(intents, intents_json['intents'])
            

        # breakfast recommendation  =================================================================================================
        def get_Breakfastfood_recommendations():
            # Query MongoDB to retrieve vegan food items
            BF_foods_cursor = db.collection.find({"mostPrefMeal": "Breakfast"})

            # Extract vegan food names from the cursor
            BF_food_items = [food_item['name'] for food_item in BF_foods_cursor]

            return BF_food_items

        if intent['intent'] == 'bf.food.recommendation':
            print("Detected 'bf.food.recommendation' intent")
            # Convert the user's message to lowercase for case-insensitive matching
            user_message = msg.lower()
            print(user_message)

            # Example usage

            if "Breakfast" or "breakfast" in user_message:
                BF_recommendations = get_Breakfastfood_recommendations()
                if BF_recommendations:
                    # Initialize an empty list to store formatted food items
                    foods_list = []
                    for food_item in BF_recommendations:
                        # Format the food item with a bullet point and add it to the list
                        formatted_food = "- " + food_item
                        foods_list.append(formatted_food)

                    # Join the formatted food items into a single string
                    foods = "\n".join(foods_list)

                    # Construct the response message
                    response = f"Here are some Breakfast food recommendations:\n{foods}"
                else:
                    response = getResponse(intents, intents_json['intents'])

        # lunch recommendation  =================================================================================================
        def get_Lunch_recommendations():
            # Query MongoDB to retrieve vegan food items
            Lunch_foods_cursor = db.collection.find({"mostPrefMeal": "Lunch"})

            # Extract vegan food names from the cursor
            Lunch_food_items = [food_item['name'] for food_item in Lunch_foods_cursor]

            return Lunch_food_items

        if intent['intent'] == 'lunch.food.desc':
            print("Detected 'lunch.food.desc' intent")
            # Convert the user's message to lowercase for case-insensitive matching
            user_message = msg.lower()
            print(user_message)

            # Example usage

            if "Lunch" or "lunch" in user_message:
                Lunch_recommendations = get_Lunch_recommendations()
                if Lunch_recommendations:
                    # Initialize an empty list to store formatted food items
                    foods_list = []
                    for food_item in Lunch_recommendations:
                        # Format the food item with a bullet point and add it to the list
                        formatted_food = "- " + food_item
                        foods_list.append(formatted_food)

                    # Join the formatted food items into a single string
                    foods = "\n".join(foods_list)

                    # Construct the response message
                    response = f"Here are some Lunch food recommendations:\n{foods}"
                else:
                    response = getResponse(intents, intents_json['intents'])

        # evening recommendation  =================================================================================================
        def get_eve_recommendations():
            # Query MongoDB to retrieve vegan food items
            eve_foods_cursor = db.collection.find({"mostPrefMeal": "Evening"})

            # Extract vegan food names from the cursor
            eve_food_items = [food_item['name'] for food_item in eve_foods_cursor]

            return eve_food_items

        if intent['intent'] == 'eve.food.desc':
            print("Detected 'eve.food.desc' intent")
            # Convert the user's message to lowercase for case-insensitive matching
            user_message = msg.lower()
            print(user_message)

            # Example usage

            if "Evening" or "evening" in user_message:
                Eve_recommendations = get_eve_recommendations()
                if Eve_recommendations:
                    # Initialize an empty list to store formatted food items
                    foods_list = []
                    for food_item in Eve_recommendations:
                        # Format the food item with a bullet point and add it to the list
                        formatted_food = "- " + food_item
                        foods_list.append(formatted_food)

                    # Join the formatted food items into a single string
                    foods = "\n".join(foods_list)

                    # Construct the response message
                    response = f"Here are some Evening snack recommendations:\n{foods}"
                else:
                    response = getResponse(intents, intents_json['intents'])

        # dinner recommendation  =================================================================================================
        def get_Dinner_recommendations():
            # Query MongoDB to retrieve vegan food items
            Dinner_foods_cursor = db.collection.find(
                {"mostPrefMeal": "Dinner"})

            # Extract vegan food names from the cursor
            Dinner_food_items = [food_item['name'] for food_item in Dinner_foods_cursor]

            return Dinner_food_items

        if intent['intent'] == 'dinner.food.desc':
            print("Detected 'dinner.food.desc' intent")
            # Convert the user's message to lowercase for case-insensitive matching
            user_message = msg.lower()
            print(user_message)

            # Example usage

            if "Dinner" or "dinner" in user_message:
                Dinner_recommendations = get_Dinner_recommendations()
                if Dinner_recommendations:
                    # Initialize an empty list to store formatted food items
                    foods_list = []
                    for food_item in Dinner_recommendations:
                        # Format the food item with a bullet point and add it to the list
                        formatted_food = "- " + food_item
                        foods_list.append(formatted_food)

                    # Join the formatted food items into a single string
                    foods = "\n".join(foods_list)

                    # Construct the response message
                    response = f"Here are some Dinner food recommendations:\n{foods}"
                else:
                    response = getResponse(intents, intents_json['intents'])

        # total price  =================================================================================================
        if intent['intent'] == 'tot_prices':
            print("Detected 'tot_prices' intent")
            # Convert the user's message to lowercase for case-insensitive matching
            user_message = msg.lower()
            print(user_message)

            food_items = db.collection3.find({}, {"oId": 1})
            food_list = [food["oId"] for food in food_items]

            # Try to find a food name in the user's message
            found_foodID = None
            for food_ID in food_list:
                if food_ID.lower() in user_message:
                    found_foodID = food_ID
                    break  # Exit the loop as soon as a match is found

            print(found_foodID)

            if found_foodID:
                # Retrieve the price of the found food item from MongoDB
                food_ID = db.collection3.find_one({"oId": found_foodID})
                if food_ID:
                    price = food_ID["subTotal"]
                    response = f"The total price of order ID: {found_foodID} is Rs.{price}0/=."

                else:
                    response = getResponse(intents, intents_json['intents'])

        if intent['intent'] == 'fallback':
            response = getResponse(intents, intents_json['intents'])
        

    # If no specific response was generated, use the default response
    if response is None:
        response = getResponse(intents, intents_json['intents'])

    return response
