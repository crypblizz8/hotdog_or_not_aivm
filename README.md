# HotDog SlackBot

Based on the [Hot Dog or Not](https://www.youtube.com/watch?v=vIci3C4JkL0&themeRefresh=1) app from the TV Show Silicon Valley.

We use

- Local ai-vm
- Run the model on a large hotdog or not / food dataset
- Then a local NGROK

AIVM usage

- Lenet5 Model
- Uploaded custom model trained on the dataset
- Then running the predictions from model
- Accuracy: 60ish%

## Pre-requsites

- You need a `kaggle.json` setup in order to interact with the files
- NGROK
- Slack Bot API KEY
- Enable these permissions on the bot in `EVENT SUBSCRIPTIONS`
- Also disabled `SOCKET MODE` for Slack for the local development process.

![SlackBotPermissions](https://github.com/user-attachments/assets/fab1112b-8022-4e87-a396-6837b2a4b7d9)

## Usage

**Files**

- `bot.py` to run the code
- `fine-tuning-lenet5.ipynb` if you want to fine tune the model further.

**Installation**

1. Clone this repo
2. Create a virtual environment to run Python `python3 -m venv .venv`
3. Linux / Mac: `source .venv/bin/activate` and Windows: `.\venv\Scripts\activate`
4. `pip install .`

**Running it**
You will have to use 3 seprate terminal windows to make it work.

1. Run AIVM by running `aivm-devnet` and wait for all of the ProxyServer to start listening
2. Run `python3 bot.py` in your venv to run the bot
3. Then run `ngrok http 8080` to run NGROK on a local port
4. In the NGROK terminal, you should have a forwarding URL like: `https://XXXX-XXX-XXX-XXX-XXX.ngrok-free.app`. Copy and paste this into the Request URL and ensure it pings your NGROK server + passes the check
5. Interact with your SlackBot by `@` the bot with a file of a hotdog or not and you should receive a `200` code and a reply.

Feel free to reach out on [Github Discussions](https://github.com/orgs/NillionNetwork/discussions) if any issues.
