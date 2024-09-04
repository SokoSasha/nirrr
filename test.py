from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from lstm_model import BestModelEverLOL
from text_processor import LanguageModel


# positive_file = "TrainingDataPositive.txt"
# negative_file = "TrainingDataNegative.txt"
# max_sequence_length = 100


def get_tokenizer(positive_file, negative_file):
    def load_data(positive_file, negative_file):
        with open(positive_file, 'r', encoding='utf-8') as f:
            positive_reviews = [line.strip() for line in f.readlines()]

        with open(negative_file, 'r', encoding='utf-8') as f:
            negative_reviews = [line.strip() for line in f.readlines()]

        return positive_reviews + negative_reviews

    reviews = load_data(positive_file, negative_file)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(reviews)

    return tokenizer


def process_review(review, model: BestModelEverLOL, tokenizer, word2vec_model, max_sequence_length):
    def preprocess_text(_reviews):
        return [review.lower().strip() for review in _reviews]

    def text_to_sequences(_reviews, _tokenizer, _max_sequence_length):
        sequences = _tokenizer.texts_to_sequences(_reviews)
        _sequences_padded = pad_sequences(sequences, maxlen=_max_sequence_length)
        return _sequences_padded

    processed_review = preprocess_text([review])
    sequences_padded = text_to_sequences(processed_review, tokenizer, max_sequence_length)

    prediction = model.predict(sequences_padded)

    confidence = prediction[0][0]
    return confidence


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
                confidence = model.predict(preprocessed_sentence, batch_size=1)[0][0]
                print(f"{confidence * 100:.2f}% positive")

        print('-----------------------------------------------------------------------------')
        model.reset_state()


if __name__ == '__main__':
    main()
