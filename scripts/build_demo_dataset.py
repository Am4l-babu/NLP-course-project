"""Build data/demo_feedback.csv.

ALL sentences below are SYNTHETIC, written by hand for demonstration and smoke
testing. They are not genuine student or customer feedback. Every feedback_id
starts with "DEMO-" so synthetic rows can never be mistaken for real ones.

The dataset is balanced: 10 categories x 3 sentiments x 10 sentences = 300 rows.
Dates are pseudo-random (fixed seed) inside a 13-week window so the dashboard's
time-based charts have something to show.

Run:  python scripts/build_demo_dataset.py
"""
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

DATA = {
    "Academics": {
        "Positive": [
            "The course content was well structured and easy to follow.",
            "I really enjoyed the new elective subjects offered this semester.",
            "The internal assessments were fair and helped me prepare for the final exam.",
            "The syllabus is up to date and relevant to current industry needs.",
            "Assignments this semester improved my understanding of the topics.",
            "The revision classes before exams were very helpful.",
            "Project-based learning in this course made the subject interesting.",
            "Great balance between theory and practical coursework this term.",
            "The academic calendar was planned well and deadlines were reasonable.",
            "The mini project gave us useful hands-on experience.",
        ],
        "Neutral": [
            "The semester examinations start on the 12th of next month.",
            "Internal assessment marks will be uploaded on the portal by Friday.",
            "The syllabus for the third semester includes six theory courses.",
            "Students must maintain seventy-five percent attendance to appear for exams.",
            "The elective registration form is available on the department notice board.",
            "The timetable for the odd semester has been published.",
            "Assignment submissions are collected every second Monday.",
            "The course has four modules and one mini project.",
            "Revaluation requests can be made within ten days of result publication.",
            "The department conducts a bridge course in the first week of the semester.",
        ],
        "Negative": [
            "The syllabus is outdated and does not match what the industry expects.",
            "Too many assignments were given with deadlines that were unrealistic.",
            "The exam portions were not covered in class at all.",
            "Internal marks were not explained and nobody told us how they were calculated.",
            "The timetable is badly planned with long gaps between classes.",
            "The question paper was far too difficult compared with what was taught.",
            "Course materials are missing for most subjects and we have to depend on friends.",
            "Results were delayed for weeks and the revaluation process is confusing.",
            "There is too much theory and almost no practical exposure in this course.",
            "The workload is overwhelming and the schedule leaves no time to revise.",
        ],
    },
    "Faculty": {
        "Positive": [
            "The faculty explained the concepts very clearly.",
            "My professor is always approachable and answers every doubt patiently.",
            "The teachers go out of their way to support weak students.",
            "Lectures are engaging and the staff use real-life examples.",
            "Our mentor gave excellent guidance on my project.",
            "The department faculty are knowledgeable and encouraging.",
            "I appreciate how the lecturer gives constructive feedback on our work.",
            "The new assistant professor teaches with great enthusiasm.",
            "Faculty members are punctual and well prepared for every class.",
            "The teacher made a difficult subject easy to understand.",
        ],
        "Neutral": [
            "The faculty meeting with students is scheduled for Thursday afternoon.",
            "A new professor will handle the data structures course this semester.",
            "Office hours for staff are from two to four in the afternoon.",
            "A guest lecturer from industry will visit the department next week.",
            "The class advisor will announce the mentor allocation list.",
            "Faculty members are required to submit the course plan before the semester begins.",
            "The staff room is on the second floor of the main block.",
            "The lecturer posted the lecture slides on the learning portal.",
            "Two new faculty members joined the department this year.",
            "Teachers will take attendance at the start of each class.",
        ],
        "Negative": [
            "The lecturer reads from slides and never explains anything properly.",
            "Some teachers are rude and refuse to clear our doubts.",
            "Classes are often cancelled without any prior notice by the staff.",
            "The faculty are not punctual and arrive late for most lectures.",
            "The professor teaches too fast and nobody can follow the lessons.",
            "Our mentor never responds to messages or meeting requests.",
            "Staff behaviour towards students is biased and unfair.",
            "The teaching method is boring and does not encourage participation.",
            "Faculty do not give feedback on assignments even after months.",
            "The teacher ignored our requests to repeat the difficult topics.",
        ],
    },
    "Infrastructure": {
        "Positive": [
            "The new classrooms are spacious, bright and well ventilated.",
            "The campus is clean and beautifully maintained.",
            "I like the new seating arrangement and the comfortable benches in the seminar hall.",
            "The auditorium has excellent acoustics and modern equipment.",
            "The renovated hostel rooms are comfortable and well kept.",
            "Drinking water stations are available on every floor, which is very convenient.",
            "The new parking area is well organised and safe.",
            "Air conditioning in the seminar hall works perfectly.",
            "The ramps and lifts make the building accessible for everyone.",
            "The sports ground is in great condition.",
        ],
        "Neutral": [
            "The seminar hall can accommodate about two hundred students.",
            "The new academic block will be opened next semester.",
            "The campus has three entry gates and a visitor parking area.",
            "Water supply will be interrupted on Sunday for maintenance work.",
            "The hostel is located about one kilometre from the main building.",
            "Classrooms on the first floor are allotted to the second year.",
            "The auditorium is booked for the annual day rehearsals this week.",
            "Construction work on the east wing will continue until December.",
            "The main building has four floors and two lifts.",
            "The sports ground is open from six in the morning to six in the evening.",
        ],
        "Negative": [
            "The classroom projector is not working.",
            "The ceiling fans in room 204 are broken and the room is unbearably hot.",
            "There is water leaking from the roof whenever it rains.",
            "The washrooms on the second floor are dirty and have no water.",
            "The lights in the corridor keep flickering and some are completely dead.",
            "The benches are broken and uncomfortable for long lectures.",
            "The lift has been out of order for two weeks.",
            "The staircase railing is loose and it feels unsafe.",
            "There is not enough parking space and vehicles block the road.",
            "Classrooms are overcrowded and many students have to stand.",
        ],
    },
    "Laboratory": {
        "Positive": [
            "The computer lab has excellent systems and enough machines for everyone.",
            "Lab instructors guide us patiently through each experiment.",
            "The chemistry laboratory is well equipped with modern apparatus.",
            "I enjoyed the hands-on sessions in the electronics lab.",
            "The lab manuals are clear and the experiments are well planned.",
            "Lab equipment is always in good working condition.",
            "The physics lab sessions helped me understand the theory better.",
            "Safety instructions in the laboratory are explained properly before every session.",
            "The new simulation software in the lab is very useful.",
            "Lab assistants are helpful and always available.",
        ],
        "Neutral": [
            "The laboratory is open until 5 PM on weekdays.",
            "Lab records must be submitted before the practical exam.",
            "Each lab batch has fifteen students.",
            "The microprocessor lab sessions are held on Wednesday mornings.",
            "Students need to wear lab coats during chemistry practicals.",
            "The lab exam will be conducted in two batches.",
            "The electronics lab is located on the ground floor of the workshop block.",
            "The lab timetable for the next week has been updated on the notice board.",
            "Students must book computer lab slots through the department office.",
            "The laboratory inventory check is scheduled for Saturday.",
        ],
        "Negative": [
            "Most of the computers in the lab are not working properly.",
            "The lab equipment is old and many instruments give wrong readings.",
            "There are too few kits so six students have to share one set.",
            "Chemicals in the lab are not stored safely and the labels are missing.",
            "The lab is always crowded and we cannot complete the experiments in time.",
            "The oscilloscope in the electronics lab has been broken for a month.",
            "Lab software is not installed and the systems keep crashing.",
            "The lab assistant is never present and we cannot get any help.",
            "The fume hood in the chemistry lab does not work, which is dangerous.",
            "Lab sessions are cancelled often without any replacement.",
        ],
    },
    "Administration": {
        "Positive": [
            "The office staff processed my certificate request very quickly.",
            "The administration was supportive and resolved my fee issue on the same day.",
            "The scholarship process was smooth and clearly explained.",
            "I appreciate the helpful attitude of the admission office.",
            "The online portal makes it easy to access notices and results.",
            "The administrative staff are courteous and well informed.",
            "Bonafide certificates are issued promptly these days.",
            "The new token system at the office has reduced waiting time.",
            "The principal's office listened to our concerns and acted on them.",
            "Fee payment through the online portal worked without any problem.",
        ],
        "Neutral": [
            "The administrative office is open from nine to four on working days.",
            "The last date for fee payment is the fifteenth of this month.",
            "Students can collect their ID cards from the front office.",
            "The scholarship application form is available on the college website.",
            "A circular about the holiday schedule was issued yesterday.",
            "Transfer certificate requests are accepted at counter number three.",
            "The office will remain closed on the second Saturday of each month.",
            "The admission process begins in June.",
            "Documents must be submitted in duplicate to the office.",
            "The college council meets once every month.",
        ],
        "Negative": [
            "The office staff are rude and keep sending us from one counter to another.",
            "My scholarship application has been pending for months with no response.",
            "The administration ignores student complaints and never replies to emails.",
            "The fee portal keeps failing and I was charged twice.",
            "Getting a simple certificate takes weeks and several visits to the office.",
            "There is no transparency in how the hostel rooms are allotted.",
            "The office is closed during lunch and the queue is endless afterwards.",
            "Notices are put up at the last minute so nobody knows about changes.",
            "Nobody in the administration could explain the extra charges on my fee receipt.",
            "Our ID cards have not been issued even after three months.",
        ],
    },
    "Library": {
        "Positive": [
            "The library has an excellent collection of reference books.",
            "The reading room is quiet and comfortable for studying.",
            "Librarians are friendly and help us find the right books.",
            "The digital library gives access to a wide range of journals.",
            "I like that the library stays open late during exam weeks.",
            "The new arrivals section keeps the collection fresh and up to date.",
            "Borrowing and returning books is quick with the new scanner system.",
            "The library has plenty of seats and charging points.",
            "The e-resources are easy to access from the hostel.",
            "Photocopy and printing facilities in the library are convenient.",
        ],
        "Neutral": [
            "The library is open from 8 AM to 6 PM on weekdays.",
            "Students can borrow up to three books for fourteen days.",
            "The library is located on the first floor of the central block.",
            "Overdue books are charged at one rupee per day.",
            "The library membership card must be shown at the entrance.",
            "The digital library requires a login through the student portal.",
            "New journals will be added to the periodicals section next month.",
            "The reference section books cannot be taken out of the library.",
            "The library will be closed on public holidays.",
            "A book exhibition is being held in the library this week.",
        ],
        "Negative": [
            "The library does not have the latest edition of the prescribed textbooks.",
            "There are very few copies of the important books, so they are always unavailable.",
            "The reading room is noisy and nobody maintains silence.",
            "The library closes too early and is shut during exam weeks.",
            "The librarian was unhelpful and rude when I asked about a missing book.",
            "The e-journal subscription has expired and we cannot access any papers.",
            "The library is stuffy, with broken fans and not enough seats.",
            "The book return process is slow and the catalogue is not updated.",
            "Books are in poor condition with torn pages and missing chapters.",
            "The photocopier in the library has not worked for weeks.",
        ],
    },
    "Canteen": {
        "Positive": [
            "The food in the canteen is tasty and freshly prepared.",
            "I like the variety of healthy options on the canteen menu.",
            "Canteen prices are reasonable for students.",
            "The canteen staff are friendly and serve quickly.",
            "The new snack counter is a great addition to the campus.",
            "The canteen is clean and the tables are well maintained.",
            "The lunch meals are filling and good value for money.",
            "The juice stall serves really fresh drinks.",
            "Portions in the canteen are generous.",
            "I appreciate that the canteen now offers vegetarian and non-vegetarian options.",
        ],
        "Neutral": [
            "The canteen is open from 8 AM to 5 PM.",
            "The canteen menu changes every week.",
            "Students can pay at the canteen using cash or UPI.",
            "The canteen is located behind the administrative block.",
            "Lunch is served between twelve thirty and two in the afternoon.",
            "A new contractor will manage the canteen from next month.",
            "The canteen has seating for about one hundred students.",
            "Tea and snacks are available at the counter in the evening.",
            "The canteen committee meets on the last Friday of the month.",
            "Meal coupons are sold at the canteen counter.",
        ],
        "Negative": [
            "The food in the canteen is cold, tasteless and overpriced.",
            "The canteen is unhygienic and flies are everywhere.",
            "Long queues at the canteen mean we miss the next class.",
            "The canteen ran out of food by 1 PM again today.",
            "The quality of the food has dropped badly and portions are tiny.",
            "I found an insect in my meal at the canteen.",
            "The tea is watery and the snacks are stale.",
            "Canteen staff are careless and charge more than the displayed prices.",
            "The canteen tables are sticky and the floor is dirty.",
            "After eating at the canteen several students felt sick.",
        ],
    },
    "Internet and Wi-Fi": {
        "Positive": [
            "The Wi-Fi connection on campus is fast and reliable.",
            "Internet speed in the library is excellent for downloading papers.",
            "The new Wi-Fi routers in the hostel work really well.",
            "I can attend online classes without any connection drops now.",
            "Getting a Wi-Fi login was quick and simple.",
            "The campus network covers the whole building, even the corridors.",
            "The IT team fixed my connection problem within an hour.",
            "Wi-Fi in the computer lab is stable even when many students are online.",
            "The internet is fast enough to stream lecture recordings.",
            "The upgraded bandwidth made online submissions hassle free.",
        ],
        "Neutral": [
            "The Wi-Fi password is changed at the beginning of every semester.",
            "Internet access is available from 6 AM to midnight in the hostel.",
            "Students can register a device for Wi-Fi at the IT help desk.",
            "The network will be down for maintenance on Sunday morning.",
            "Each student is allowed to connect two devices to the campus network.",
            "The IT department is located on the second floor of the main block.",
            "A new Wi-Fi policy will be shared through email.",
            "Wi-Fi coverage includes the library, the hostel and the seminar halls.",
            "The campus internet is provided by two different service providers.",
            "The Wi-Fi login page appears after connecting to the college network.",
        ],
        "Negative": [
            "The Wi-Fi keeps disconnecting every few minutes.",
            "Internet speed in the hostel is painfully slow in the evenings.",
            "There is no Wi-Fi signal at all in the classrooms on the third floor.",
            "I cannot log in to the campus network and the IT desk does not respond.",
            "The internet goes down during online exams and we lose our work.",
            "The Wi-Fi is blocked for most educational websites.",
            "Network speed drops to almost nothing when many students connect.",
            "The Wi-Fi password was changed again without informing anyone.",
            "Downloading even a small file takes ages on the campus network.",
            "The broadband in the computer lab is down again.",
        ],
    },
    "Transport": {
        "Positive": [
            "The college bus is always on time and the drivers are careful.",
            "The new bus route covers my area very conveniently.",
            "Bus fares are affordable and the seats are comfortable.",
            "The transport office arranged an extra bus for the exam days.",
            "I feel safe travelling in the college bus.",
            "The bus staff are polite and helpful.",
            "Pick-up and drop timings suit our class schedule well.",
            "Buses are clean and well maintained.",
            "The shuttle service between the hostel and the campus is very handy.",
            "The tracking app helps me know exactly when the bus will arrive.",
        ],
        "Neutral": [
            "The college bus leaves from the main gate at 4:30 PM.",
            "Bus passes must be renewed at the start of each semester.",
            "There are six bus routes covering the city and nearby towns.",
            "The transport office is next to the security cabin.",
            "The bus fare is collected monthly through the fee portal.",
            "A new bus timetable will be displayed on the notice board.",
            "Students can request a change of bus stop by filling out a form.",
            "The shuttle runs every thirty minutes between the hostel and the campus.",
            "Bus number five takes the highway route.",
            "Transport services will not operate on holidays.",
        ],
        "Negative": [
            "The college bus is always late and we miss the first class.",
            "Buses are overcrowded and students have to stand for an hour.",
            "The driver drives rashly and it is unsafe for passengers.",
            "The bus broke down on the way and nobody informed us.",
            "There is no bus service on our route in the evening.",
            "The seats in the bus are torn and the windows will not close.",
            "The bus fare was increased without any notice.",
            "The bus skipped our stop again and the transport office ignored our complaint.",
            "The return bus leaves too early, before the last class ends.",
            "Buses are dirty and smell terrible.",
        ],
    },
    "Other": {
        "Positive": [
            "The annual cultural fest was wonderfully organised.",
            "Thank you for arranging the career guidance session.",
            "The placement training programme was very useful.",
            "I enjoyed the sports day and the friendly atmosphere on campus.",
            "The counselling service is very supportive for students.",
            "The alumni meet was a great opportunity to network.",
            "The NSS volunteer programme was a rewarding experience.",
            "The college events keep students motivated and engaged.",
            "Great initiative to organise the blood donation camp.",
            "The student clubs offer wonderful opportunities to learn new skills.",
        ],
        "Neutral": [
            "The annual day celebration will be held in the auditorium.",
            "The placement drive registration closes next Friday.",
            "A blood donation camp is scheduled for the 20th of this month.",
            "Students can join clubs through the activities coordinator.",
            "The alumni association meets twice a year.",
            "The sports meet begins on Monday.",
            "The NSS unit conducts a cleaning drive every semester.",
            "A list of upcoming events is available on the notice board.",
            "The counselling room is open on Tuesdays and Thursdays.",
            "The college festival committee will hold its first meeting tomorrow.",
        ],
        "Negative": [
            "The cultural fest was poorly organised and started two hours late.",
            "The placement training sessions were a waste of time.",
            "Nobody told us about the sports day until the last minute.",
            "The counsellor is rarely available and no appointments are given.",
            "There is no support for students who want to start clubs.",
            "Event announcements are confusing and keep changing.",
            "The security at the gate is careless and strangers enter the campus freely.",
            "Students are not informed about placement opportunities in time.",
            "The college does not respond to harassment complaints.",
            "Too many events are held during the exam period.",
        ],
    },
}


def build() -> pd.DataFrame:
    rng = random.Random(config.RANDOM_SEED)
    rows = []
    for category, by_sentiment in DATA.items():
        assert category in config.CATEGORIES, category
        for sentiment, sentences in by_sentiment.items():
            assert sentiment in config.SENTIMENTS, sentiment
            for text in sentences:
                rows.append({"feedback_text": text, "sentiment": sentiment, "category": category})
    rng.shuffle(rows)
    start = datetime(2026, 1, 5)
    for i, row in enumerate(rows, start=1):
        day = start + timedelta(days=rng.randrange(0, 91))
        row["feedback_id"] = f"DEMO-{i:04d}"
        row["created_at"] = day.strftime("%Y-%m-%d")
    df = pd.DataFrame(rows)[["feedback_id", "feedback_text", "sentiment", "category", "created_at"]]
    assert df["feedback_text"].is_unique, "duplicate sentences in demo data"
    return df


if __name__ == "__main__":
    df = build()
    config.DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(config.DEMO_DATA_PATH, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} synthetic demo rows to {config.DEMO_DATA_PATH}")
    print(df.groupby(["category", "sentiment"]).size().unstack().to_string())
