from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()

def main():
    print("Hello from langchain!")

    information = """

Spider-Man is a superhero in American comic books published by Marvel Comics. Created by writer-editor Stan Lee and artist Steve Ditko, he first appeared in the anthology comic book Amazing Fantasy #15 (August 1962) in the Silver Age of Comic Books. Widely regarded as one of the most popular and commercially successful superheroes, he has been featured in comic books, television shows, films, video games, novels, and plays.

Spider-Man's secret identity is Peter Benjamin Parker, who was raised by his Aunt May and Uncle Ben in Queens, New York City, after the death of his parents. Lee, Ditko, and later writers had the character deal with the struggles of adolescence and young adulthood. Readers identified with his self-doubt and loneliness. Unlike previous teen heroes, Spider-Man would be given many supporting characters. These include his Daily Bugle boss, J. Jonah Jameson; friends Harry Osborn, the Human Torch, and Flash Thompson; romantic interests Gwen Stacy, Mary Jane Watson, and the Black Cat; and enemies Doctor Octopus, the Green Goblin, and Venom. In his origin story, Peter gets his superhuman spider-powers and abilities after he was bitten by a radioactive spider. These powers include superhuman strength, speed, agility, reflexes and durability; clinging to surfaces and ceilings; and detecting danger with his precognitive "spider-sense". He sews a spider-web patterned spandex costume that fully covers his body and builds wrist-mounted "web-shooter" devices that shoot artificial spider-webs of his own design, which he uses for both fighting and "web swinging" across the city. Peter initially used his powers for personal gain, but after his Uncle Ben was killed by a burglar that he could have stopped but chose not to, he learned that "with great power comes great responsibility", and began to use his powers to fight crime as Spider-Man.

Marvel has featured Spider-Man in several comic book series, the first and longest-lasting of which is The Amazing Spider-Man. Since his introduction, the main-continuity version of Peter has gone from a high school student to attending college to currently being somewhere in his late 20s. Peter has been a member of numerous superhero teams, most notably the Avengers and Fantastic Four. Doctor Octopus also took on the identity for a story arc spanning 2012–2014 following the "Dying Wish" storyline, where Peter appears to die after Doctor Octopus orchestrates a body swap with him and becomes the Superior Spider-Man. Marvel has also published comic books featuring alternate versions of Spider-Man, including Spider-Man 2099, which features the adventures of Miguel O'Hara, the Spider-Man of the future; Ultimate Spider-Man, which features the adventures of a teenage Peter Parker in the alternate universe; and Ultimate Comics: Spider-Man, which depicts a teenager named Miles Morales who takes up the mantle of Spider-Man after Ultimate Peter Parker's apparent death. Miles later became a superhero in his own right and was brought into mainstream continuity during the Secret Wars event, where he sometimes works alongside the mainline version of Peter.

Spider-Man has appeared in many forms of media, including several animated TV series, a live-action television series, syndicated newspaper comic strips, and multiple series of films. In live-action films, Spider-Man has been portrayed by Tobey Maguire in Sam Raimi's Spider-Man trilogy, Andrew Garfield in The Amazing Spider-Man duology directed by Marc Webb, and Tom Holland in the Marvel Cinematic Universe. The Peter Parker version of Spider-Man was also voiced by Jake Johnson and Chris Pine in the animated film Spider-Man: Into the Spider-Verse, with the former reprising his role in the sequel, Spider-Man: Across the Spider-Verse.

"""

    summary_template = """

        When saying any answer to the user's query {{information}} always say it in a nice and simple and attractive way about the super hero.

    """

    summary_prompt_template = PromptTemplate(
        input_variables = ['information'], template = summary_template
    )

    llm = ChatOllama(temperature=0.6, model='qwen3:8b')
    chain = summary_prompt_template | llm
    response = chain.invoke(input={'information':information})
    print(response.content)
    # print(os.getenv("LANGSMITH_TRACING"))
    # print(bool(os.getenv("LANGSMITH_API_KEY")))
    # print(os.getenv("LANGSMITH_PROJECT"))

if __name__ == "__main__":
    main()
