import time

import matplotlib.pyplot as plt

from lstm_model import BestModelEverLOL
from text_processor import LanguageModel
from tensorflow.keras.preprocessing.sequence import pad_sequences


def main():
    model = BestModelEverLOL.load()
    lm = LanguageModel.load()
    msl = model.get_max_sequence_length

    # Список отзывов для обработки
    reviews = [
        "Acadamy of Scuba is great for new or old divers. I love this place. I found AOS in October 2012 when I moved back to Phoenix. I remember driving by this place years ago and always wanting to stop in, but I never did. Evidently, AOS opened in this long-time scuba spot in 2012. I first came here in October 2012 to take the Rescue Diver class. The class was great. It was taught by CG Durda, and it really was thorough. At AOS, the classes-especially the advanced classes-aren't just about accomplishing the bare minimum. They usually add tons of extra activities so that you are really learning and benefiting from the instructors' extensive experience. Most of the instructors have hundreds or thousands of dives and have dove in all sorts of conditions. AOS's model is \"the dive shop that dives,\" and that's really true. Everyone who works there considers scuba diving their passion. When we had our open water dives for Resuce, there were tons of people out at Lake Pleasant just diving or hanging out between dives. AOS creates a community for divers, and anyone can become a part of it. As an outsider, I was immediately made to feel welcome, and I was told all about the meet-ups out at the lake for diving. If you don't have a buddy, there is always someone you can join. Continuing on the social note, AOS has two monthly socials a week. I typically am pretty introverted and don't partake in such thing, but I made myself go to one-and now I go all the time. Why Because now I have a ton of friends in the dive community, and this is a regular way to meet up, talking diving, or just hang out. There are all sorts of people in this crowd-from heart surgeons to construction workers-but the constant between everyone is friendliness and a love for diving. AOS also has tons of trips, and what is great about that is it isn't just a ticket clearance office, but a hub for dive trips that puts together lots of Phoenix divers. In April 2013, I've already booked a trip to Roatan, where the entire resort will be divers from Phoenix. Since I've been diving with AOS now for a few months, I know a ton of people on the trip. Plus, all the dive leaders will be staff I've gotten to know and are comfortable with. On the learning side, I purchased their \"complete diver\" program, which is a ridiculously affordably priced package for multiple dive specialties. Since July, I've complete several additional specialties, such as dry suit, night diver, debris removal diver, fish identification, equipment specialist, underwater navigator, and DAN's DEMP course (which is a super comprehensive emergency first responderO2 providerfirst aid course). The instructors I've had include Dominic Diodato, John Flanders (owner), and Randall Medders, as well as a host of others who drop in to help out. All of ths instructors are incredibly patient and knowledgeable. There is no attitude of superiority here-even amongst people like John, who has had thousands of dives. I'm not naming all of their staff, but I have had experiences with most of them, and, while all have their different strengths and weaknesses, there isn't a bad one among them. There is always people hanging around the shop, and the indoor pool is often open just to come in and practice skills. The selection of gear is good, and what they don't have at the store they can order for you. Prices are a little higher than online, but I'm ok with paying a little more to have a local store to check things out. I think they have price match, though. In talking with the owner, I know that there is a method to the way they stock their store. For example, while there aren't tons and tons of wetsuits always sitting around, that is because wet suite degrade, and they won't sell a customer a wetsuit that's been sitting on the shelf a year and lost some of its insulating abiltiies. ASK if you want something you don't see. Both Dominic at the Metro location and Tania at the Paradise Valley location are very accomodating. I've had an opportunity also to observe open water classes and discover scuba diving. Personally, I've referred several of my own friends to AOS. They are extremely thorough, never compromise PADI standards, and, as always, add additional experiences to enhance the class. The instructors are patient and will work with any skill level, age, or fear. I know they have people specialized to help students with physical limitations, for example, and they work with a lot of kids, as well. I've always been pretty picky about what dive shop I use, because I want to be able to trust the people helping equip and train me for what can be a hazardous environment. AOS has never let me down. And more than that, I've met a ton of people that have become my friends. I've always loved diving since I first dove 15 years ago, but I was always a once or twice a year vacation diver. AOS has allowed my to develop my passion through continuing education and being a member of an active dive community.",
        "The location is in a strip mall, but this place is a diamond in the rough.  The food was some of the best italian food I have had anywhere.  I live in Kansas City, which has a pretty large and long standing italian populaiton.  The food is a good as any of the old italian places in KC.  Everyone in my group was amazed at the quality of the food for the price.  I will keep this one on my list of places to come back to when I am in AZ again.",
        "I live in Kansas City, which has a pretty large and long standing italian population. Everyone in my group was amazed at the quality of the food for the price.  I will keep this one on my list of places to come back to when I am in AZ again.",
        "Never going back. This was my second time here. Wasn't that impressed the first time, but thought I'll give it another look, more because one of the guys in our team really wanted to go here. Food: Just ok, over-priced and very small portions. Came out hungry. Deserts are quite nice. Service: I think I would have given this place 2 12 stars was it not for the bad service. Very rude waiter, I am not going to go into details of what happenned at our table but suffice to say I left the place pissed. Never going back and nor is the guy in our team who really wanted to go in the first place.I think I hate a rude (and I mean rude not bad) service more that bad food. Bad food just leaves me with no feelings but a rude server leaves me pissed.",
        "Food: Just ok, over-priced and very small portions. Deserts are quite nice. Service: I think I would have given this place 2 12 stars was it not for the bad service. Very rude waiter, I am not going to go into details of what happenned at our table but suffice to say I left the place pissed. Never going back and nor is the guy in our team who really wanted to go in the first place.I think I hate a rude (and I mean rude not bad) service more that bad food. Bad food just leaves me with no feelings but a rude server leaves me pissed.",
    ]

    warmup = pad_sequences(
        lm.texts_to_sequences(["The location is in a strip mall, but this place is a diamond in the rough."]),
        maxlen=msl)
    model.predict(warmup, verbose=0)

    total_time = 0
    total_sentences = 0

    for review in reviews:
        conf = []
        times = []
        print(f'Review: {review}')
        review = review.split('.')
        context_buffer = []
        for sentence in review:
            if sentence.strip():
                start_time = time.perf_counter()
                context_buffer.extend(lm.texts_to_sequences([sentence])[0])
                lcb = len(context_buffer)
                if lcb > msl:
                    context_buffer = context_buffer[lcb-msl:]
                sequences_padded = lm.pad_sequences([context_buffer], maxlen=msl)
                confidence = model.predict(sequences_padded, verbose=0)[0][0]
                conf.append(confidence)
                elapsed_time = time.perf_counter() - start_time
                times.append(elapsed_time)
                total_time += elapsed_time
                total_sentences += 1
                # print(f"{elapsed_time:.4f} seconds")
                # print(f"{confidence * 100:.2f}% positive")
                if confidence < 0.5:
                    print(f'{sentence}\n\t{confidence}')
        model.reset()

        plt.figure(figsize=(12, 5))

        # График уверенности
        plt.subplot(1, 2, 1)
        plt.plot(conf, marker='o', color='b')
        plt.title('Confidence Levels')
        plt.xlabel('Sentence Index')
        plt.ylabel('Confidence')
        plt.ylim(0, 1)
        plt.grid()

        # График времени
        plt.subplot(1, 2, 2)
        plt.plot(times, marker='o', color='r')
        plt.title('Processing Times')
        plt.xlabel('Sentence Index')
        plt.ylabel('Time (seconds)')
        plt.ylim(0, 0.1)
        plt.grid()

        plt.tight_layout()
        plt.show()

        print('-----------------------------------------------------------------------------')

    print(f'Average processing time per sentence: {total_time / total_sentences:.4f} seconds')


if __name__ == '__main__':
    main()
