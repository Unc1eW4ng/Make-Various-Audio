from openai import OpenAI

model = 'gpt-4o'

def get_various_caption(caption):
  api_key = ''



  client = OpenAI(
    #中转的url地址
    #修改为自己生成的key
    api_key=api_key
  )

  # caption = "a violin piece"

  completion = client.chat.completions.create(
    model=model,
    messages=[
      { "role": "user", 
        "content": f"There is  a simple description of a piece of music, please enrich it and make it more vivid.Provide 5 various version(divided with /) in five line,each no more than 20 words\
      for example:\
        question: a guitar solo\
        answer: A soothing guitar solo that will make you feel like you're on a beach./A rapid electric guitar solo/The guitar's melancholic solo notes tell an untold story, stirring profound emotions./The guitar solo, vibrates with intensity and depth./High pitch rock with exciting guitar solo \
        question: {caption}"
      }
    ]
  )

  response_content = completion.choices[0].message.content
  return response_content.split('/')


def getmodel():
  return model
