from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from lstm_model import BestModelEverLOL
from text_processor import LanguageModel


def main():
    model = BestModelEverLOL.load()
    lm = LanguageModel.load()

    # Список отзывов для обработки
    reviews = [
        "The location is in a strip mall, but this place is a diamond in the rough.  The food was some of the best italian food I have had anywhere.  I live in Kansas City, which has a pretty large and long standing italian populaiton.  The food is a good as any of the old italian places in KC.  Everyone in my group was amazed at the quality of the food for the price.  I will keep this one on my list of places to come back to when I am in AZ again.",
        "I live in Kansas City, which has a pretty large and long standing italian populaiton. Everyone in my group was amazed at the quality of the food for the price.  I will keep this one on my list of places to come back to when I am in AZ again.",
        "Never going back!!This was my second time here. Wasn't that impressed the first time, but thought I'll give it another look, more because one of the guys in our team really wanted to go here. Food: Just ok, over-priced and very small portions. Came out hungry. Deserts are quite niceService: I think I would have given this place 2 12 stars was it not for the bad service. Very rude waiter, I am not going to go into details of what happenned at our table but suffice to say I left the place pissed! Never going back and nor is the guy in our team who really wanted to go in the first place.I think I hate a rude (and I mean rude not bad) service more that bad food. Bad food just leaves me with no feelings but a rude server leaves me pissed!",
        "Food: Just ok, over-priced and very small portions. Deserts are quite niceService: I think I would have given this place 2 12 stars was it not for the bad service. Very rude waiter, I am not going to go into details of what happenned at our table but suffice to say I left the place pissed! Never going back and nor is the guy in our team who really wanted to go in the first place.I think I hate a rude (and I mean rude not bad) service more that bad food. Bad food just leaves me with no feelings but a rude server leaves me pissed!",
    ]

    for review in reviews:
        print(f'Review: {review}')
        review = review.split('.')
        for sentence in review:
            if sentence.strip():
                preprocessed_sentence = lm.preprocess([sentence])
                confidence = model.predict(preprocessed_sentence, verbose=1)[0][0]
                print(f"{confidence * 100:.2f}% positive")

        print('-----------------------------------------------------------------------------')
        model.reset_state(verbose=1)


if __name__ == '__main__':
    main()
