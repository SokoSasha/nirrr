from gensim.models import Word2Vec
from keras.src.layers import Bidirectional
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

positive_file = "TrainingDataPositive.txt"
negative_file = "TrainingDataNegative.txt"
max_sequence_length = 100


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


def process_review(review, model, tokenizer, word2vec_model, max_sequence_length):
    def preprocess_text(_reviews):
        return [review.lower().strip() for review in _reviews]

    def text_to_sequences(_reviews, _tokenizer, _max_sequence_length):
        sequences = _tokenizer.texts_to_sequences(_reviews)
        _sequences_padded = pad_sequences(sequences, maxlen=_max_sequence_length)
        return _sequences_padded

    processed_review = preprocess_text([review])
    sequences_padded = text_to_sequences(processed_review, tokenizer, max_sequence_length)

    prediction = model.predict(sequences_padded, batch_size=1, verbose=0)

    confidence = prediction[0][0]
    return confidence


def reset_lstm(model):
    for layer in model.layers:
        if isinstance(layer, LSTM) or isinstance(layer, Bidirectional):
            layer.reset_states()


def main():
    model = load_model('best_model.keras')
    word2vec_model = Word2Vec.load('word2vec_model.bin')
    tokenizer = get_tokenizer(positive_file, negative_file)

    # Список отзывов для обработки
    reviews = [
        "The location is in a strip mall, but this place is a diamond in the rough.  The food was some of the best italian food I have had anywhere.  I live in Kansas City, which has a pretty large and long standing italian populaiton.  The food is a good as any of the old italian places in KC.  Everyone in my group was amazed at the quality of the food for the price.  I will keep this one on my list of places to come back to when I am in AZ again.",
        "Fantastic spot for an even or a quite cocktail.  They were swell to host the Yelp crew with a great drink menu and super attentive staff.I'd certainly recommend anything with the purred fruit in it (apple, any of them really)!",
        "Love this place. Stiff martinis and cocktails, cheap drinks, good service, nice atmosphere to chill in the upstairs lounge and hang up with your friends. Classy crowd and much more mature, older, and professional crowd. There aren't any of those college frat boys or belligerent drunks there, which is so great. Very nice place to wind down after a long day.",
        "Never going back!!This was my second time here. Wasn't that impressed the first time, but thought I'll give it another look, more because one of the guys in our team really wanted to go here. Food: Just ok, over-priced and very small portions. Came out hungry. Deserts are quite niceService: I think I would have given this place 2 12 stars was it not for the bad service. Very rude waiter, I am not going to go into details of what happenned at our table but suffice to say I left the place pissed! Never going back and nor is the guy in our team who really wanted to go in the first place.I think I hate a rude (and I mean rude not bad) service more that bad food. Bad food just leaves me with no feelings but a rude server leaves me pissed!"
        # "After being admitted to this facility my elderly friend received a few days treatment and then was constantly badgered to sign into a hospice facility called Solari. She was told by TLC that she could stay there up to 100 days. Then suddenly, after less than 2 weeks she was told she had to leave...period, no notice. Once again the Solari rep showed up only this time she stated that she could only stay 2 days in Solari. Moved her the following day. TLC on site social worker was usually no where to be found and never returned her calls or the many pages the receptionist made for her stating ""I know she's here, I don't know why she's not responding"".Avoid this place at any costs!",
    ]

    for review in reviews:
        print(f'Review: {review}')
        review = review.split('.')
        for sentence in review:
            confidence = process_review(sentence, model, tokenizer, word2vec_model, max_sequence_length)
            if confidence > 0.5:
                print(f"{confidence * 100:.2f}% positive")
            else:
                print(f"{confidence * 100:.2f}% negative")

        print()
        reset_lstm(model)


if __name__ == '__main__':
    main()
