from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from sentence_transformers import SentenceTransformer , InputExample , util
from sentence_transformers import models, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import numpy as np
import torch
import json
import requests
from neo4j import GraphDatabase, basic_auth
import faiss
import pandas as pd
from linebot import LineBotApi
from linebot.models import (
    QuickReply, 
    QuickReplyButton, 
    MessageAction, 
    TextSendMessage, 
    CarouselTemplate, 
    FlexSendMessage ,
    BubbleContainer, 
    ImageComponent, 
    TextComponent,
    PostbackEvent,
    PostbackAction,
    BoxComponent
)

from linebot.models import FlexSendMessage, BubbleContainer, ImageComponent, TextComponent
from PIL import Image

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "1234567890")
def run_query(query, parameters=None):
   with GraphDatabase.driver(URI, auth=AUTH) as driver:
       driver.verify_connectivity()
       with driver.session() as session:
           result = session.run(query, parameters)
           return [record for record in result]
   driver.close()

def requirement_query(query, parameters=None):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()  # Check if the connection works
        with driver.session() as session:
            result = session.run(query)  # Run the Cypher query
            # Iterate through the result and return the first record
            record = result.single()  # This fetches one record if it exists
            if record:
                return record['detail'], record['reply']
            else:
                return None, None
        

cypher_query_greeting = '''
MATCH (n:Greeting) RETURN n.name as name, n.msg_reply as reply;
'''
cypher_query_end_conversation = '''
MATCH (n:ending) RETURN n.name as name, n.msg_reply as reply;
'''
cypher_query_name = '''
MATCH (n:Name) RETURN n.name as name, n.msg_reply as reply;
'''
cypher_query_requirement = '''
MATCH (n:Requirement) RETURN n.name as name, n.msg_reply as reply , n.detail as detail;
'''
cypher_query_quickreply = '''
MATCH (n:Quickreply) RETURN n.name as name, n.msg_reply as reply;
'''

greeting_corpus = []
end_corpus = []
name_corpus = []
requirement_corpus = []
quickreply_corpus = []

greeting_vec = None
end_vec = None
name_vec = None
requirement_vec = None
quickreply_vec = None

results1 = run_query(cypher_query_greeting)
for record in results1:
    greeting_corpus.append(record['name'])
greeting_corpus = list(set(greeting_corpus))
greeting_vec = model.encode(greeting_corpus)

results2 = run_query(cypher_query_end_conversation)
for record in results2:
    end_corpus.append(record['name'])
end_corpus = list(set(end_corpus))
end_vec = model.encode(end_corpus)

results3 = run_query(cypher_query_name)
for record in results3:
    name_corpus.append(record['name'])
name_corpus = list(set(name_corpus))
name_vec = model.encode(name_corpus)

results4 = run_query(cypher_query_requirement)
for record in results4:
    requirement_corpus.append(record['name'])
requirement_corpus = list(set(requirement_corpus))
requirement_vec = model.encode(requirement_corpus)

results5 = run_query(cypher_query_quickreply)
for record in results5:
    quickreply_corpus.append(record['name'])
quickreply_corpus = list(set(quickreply_corpus))
quickreply_vec = model.encode(quickreply_corpus)

session = requests.Session()
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Adjust URL if necessary
headers = {
    "Content-Type": "application/json"
}
selected_product_properties = {}
filter = []
asking = 0
summit = False
products = []
press = False

def character_reply_requirement(input , uid):
    global session , OLLAMA_API_URL,headers , selected_product_properties
    payload = { 
            # "model": "supachai/llama-3-typhoon-v1.5", 
            "model": "llama3.2", 
            # "prompt": f"""(ข้อความ) : {input} ,ช่วยเรียบเรียงข้อความนี้ใหม่แต่ใจความของข้อความยังคงเดิม กระชับสั้นๆ ไม่เกิน 30 token llama3 สวมบทบาทเป็นพนักงานขายน้ำหอมผู้ชายที่สุภาพและกระตือรือร้น ที่ตอบคำถามลูกค้า โดยเริ่มต้นด้วยประโยคที่แสดงความพร้อมในการช่วยเหลือ (ข้อความ)""",
            "prompt": f"""BOT (original message to user): [{input}] ,Please Repharsing the message .Respond as a polite male salesperson. Use 'ครับ' at the end. Do not exceed 20 tokens. Respond in Thai language only. """,
            "stream": False  , "options":{"num_ctx": 1024, "temperature": 0.8}
        } 
    response = session.post(OLLAMA_API_URL, headers=headers, data= json.dumps(payload))

    if response.status_code == 200:
        response_data = response.text 
        data = json.loads(response_data) 
        decoded_text = data["response"] 
        return  decoded_text.replace('ค่ะ','ครับ').strip()
       
    else:
        return   (f"Failed to get a response: {response.status_code}, {response.text}")
    

def character_reply_specific_product(input , uid):
    global session , OLLAMA_API_URL,headers , selected_product_properties
    # chat_history = get_chat_history(uid)
    
    # history_context = ""
    # for entry in chat_history:
    #     history_context += f"User ข้อความ: {entry['question']}\nBot ตอบกลับ: {entry['reply']}\n"
    
    # # เพิ่มคำถามใหม่ในบริบท
    # full_context = f"{history_context}\n"
    payload = { 
            # "model": "supachai/llama-3-typhoon-v1.5", 
            "model": "llama3.2", 
            "prompt": f"""สินค้า : {selected_product_properties['name']}, การตอบกลับก่อนปรับข้อความ : {input} ,ช่วยเรียบเรียงข้อความนี้ใหม่ โดยมีข้อมูลยังครบถ้วนเหมือนเดิม เป็นประโยคที่เข้าใจง่าย  ตอบเฉพาะส่วนที่เป็นคำตอบเท่านั้น โดยใช้โทนคำพูดแบบพนักงานชายที่แนะนำน้ำหอม สุภาพ พูดไพเราะ น่ารักๆ""",
            "stream": False  , "options":{"num_ctx": 1024, "temperature": 0.8}
        }
    response = session.post(OLLAMA_API_URL, headers=headers, data= json.dumps(payload))

    if response.status_code == 200:
        response_data = response.text 
        data = json.loads(response_data) 
        decoded_text = data["response"] 
        return  decoded_text.replace('ค่ะ','ครับ').strip()
       
    else:
        return   (f"Failed to get a response: {response.status_code}, {response.text}")
    

def ollama(input, reply , score ,sent , uid ):
    global session , OLLAMA_API_URL,headers
    chat_history = get_chat_history(uid)
    
    history_context = ""
    for entry in chat_history:
        history_context += f"User ข้อความ: {entry['question']}\nBot ตอบกลับ: {entry['reply']}\n"
    
    # เพิ่มคำถามใหม่ในบริบท
    full_context = f"{history_context}\n"
   
    if score < 0.6 :
        # suffix = '\n\n- เรียบเรียงการตอบกลับจาก Neo4j ด้วย Ollama \U0001F999 -'
        payload = {
            # "model": "supachai/llama-3-typhoon-v1.5", 
            "model": "llama3.2", 
            "prompt": f"""ประวัติการสนทนา :{full_context}ข้อความ : {sent} ?, การตอบกลับ : {reply} , ช่วยสร้างคำตอบกลับของข้อความนี้ใหม่ โดยมีใจความเหมือนเดิมเป็นประโยคที่เข้าใจง่าย ไม่เกิน 30 คำ ตอบเฉพาะส่วนที่เป็นคำตอบเท่านั้น โดยใช้โทนคำพูดแบบเด็กผู้ชายที่สุภาพ น่ารักๆ พร้อมแสดงความยินดีทุกครั้งที่ได้ตอบคำถาม""",
            "stream": False  , "options":{"num_ctx": 1024, "temperature": 0.8,}
        }
        question = sent
    else : 
        # suffix = '\n\n- สร้างข้อความตอบกลับด้วย Ollama \U0001F999 -'
        payload = { 
            # "model": "supachai/llama-3-typhoon-v1.5",  
            "model": "llama3.2", 
            "prompt": f"""ประวัติการสนทนา :{full_context}ข้อความ : {sent} ? ,ช่วยตอบกลับข้อความคำถามด้วยประโยคที่มีเหตุผล กระชับ ไม่เกิน 30 คำ ตอบเฉพาะส่วนที่เป็นคำตอบเท่านั้น โดยใช้โทนคำพูดแบบเด็กผู้ชายที่สุภาพ น่ารักๆ พร้อมแสดงความยินดีทุกครั้งที่ได้ตอบคำถาม""",
            "stream": False , "options":{"num_ctx": 1024, "temperature": 0.8,}
        }  
        question = sent
    response = session.post(OLLAMA_API_URL, headers=headers, data= json.dumps(payload))

    if response.status_code == 200:
        response_data = response.text 
        data = json.loads(response_data) 
        decoded_text = data["response"] 
        return  question , decoded_text.replace('ค่ะ','ครับ').strip()
       
    else:
        return   question ,(f"Failed to get a response: {response.status_code}, {response.text}")
    

def faiss_index(vector):
    vector_dimension = vector.shape[1]
    index = faiss.IndexFlatL2(vector_dimension)
    faiss.normalize_L2(vector)
    index.add(vector)
    return index

def compute_nearest(vector , sentence , corpus):
    df = pd.DataFrame(corpus, columns=['contents'])
    index = faiss_index(vector)
    search_vector = model.encode(sentence)
    _vector = np.array([search_vector])
    faiss.normalize_L2(_vector)
    k = index.ntotal
    distances , ann = index.search(_vector , k = k)
    results = pd.DataFrame({'distances':distances[0],  'ann' : ann[0]})
    merge = pd.merge(results, df ,left_on = 'ann' , right_index = True)
    return  merge['contents'][0] , merge['distances'][0] 


def neo4j_search(neo_query):
   results = run_query(neo_query)
   for record in results:
       response_msg = record['reply']
   return response_msg      


def compute_response(sentence , uid):
    global asking , summit , filter , filter_list
    greeting_word , greeting_score = compute_nearest(greeting_vec , sentence , greeting_corpus)
    end_word , end_score = compute_nearest(end_vec , sentence , end_corpus)
    name_word , name_score = compute_nearest(name_vec , sentence , name_corpus )
    requirement_word , requirement_score = compute_nearest(requirement_vec , sentence , requirement_corpus)
    quickreply_word , quickreply_score = compute_nearest(quickreply_vec , sentence , quickreply_corpus)
    print(f'Distance Greeting [{greeting_word}]: {greeting_score}')
    print(f'Distance Ending [{end_word}]: {end_score}')
    print(f'Distance Name [{name_word}]: {name_score}')
    print(f'Distance Requirement [{requirement_word}]: {requirement_score}')
    print(f'Distance Quickreply [{quickreply_word}]: {quickreply_score}')

    

    min_dis = [greeting_score , end_score , name_score , requirement_score , quickreply_score]
    matching  = [greeting_word , end_word , name_word , requirement_word , quickreply_word]
    min_index = min_dis.index(min(min_dis))
    cypher_match = ['Greeting' , 'ending' , 'Name' , 'Requirement' , 'Quickreply'] 
    print(cypher_match[min_index]) 
    print(matching[min_index])
    My_cypher = f"""MATCH (n:{cypher_match[min_index]}) where n.name ="{matching[min_index]}" RETURN n.msg_reply as reply"""
    my_msg  = neo4j_search(My_cypher)
    print(my_msg) 
    print(min(min_dis))
    print(sentence)
    if cypher_match[min_index] == 'Greeting' and min(min_dis) < 0.7:
        return sentence , my_msg , cypher_match[min_index] 
    elif cypher_match[min_index] == 'Requirement' and min(min_dis) < 0.9:
        query = f"""MATCH (n:{cypher_match[min_index]}) where n.name ="{matching[min_index]}" RETURN n.msg_reply as reply , n.detail as detail"""
        print('-==========================================----------------')
        detail , reply = requirement_query(query)
        if detail == 'การยืนยัน':
                summit = True  
                print(filter)
                return sentence , 'สักครู่ค้าบบ' , cypher_match[min_index]
        else :
                filter.append([detail , reply])
                print(filter)
                return sentence , 'ต้องการอะไรเพิ่มเติมอีกไหมครับ?',cypher_match[min_index]
    
    elif cypher_match[min_index] == 'Quickreply':
        return sentence , my_msg , cypher_match[min_index]

    else: 
        question , gen_msg = ollama(matching[min_index] , my_msg , min(min_dis) , sentence , uid)
        return question , gen_msg , cypher_match[min_index]

def quick_reply_menu(line_bot_api, tk, user_id, msg):

    # ปุ่มเมนู
    menu_button = QuickReplyButton(
        action=MessageAction(label="เมนู", text="เมนู"),
        image_url= "  https://cdn-icons-png.flaticon.com/32/7543/7543108.png ",
    )

    # ปุ่มล้างประวัติ
    clear_chat_button = QuickReplyButton(
        action=MessageAction(label="ล้างประวัติ", text="ล้างประวัติ"),
        image_url="https://example.com/path/to/menu_icon.png"
    )

    # เพิ่มปุ่มอื่นๆ ลงใน Quick Reply
    quick_reply = QuickReply(
        items=[
            menu_button,
            clear_chat_button,
            # สามารถเพิ่มปุ่มอื่นๆ ได้ตามต้องการ
        ]
    )

    return quick_reply

def quick_reply_menuu(line_bot_api, tk, user_id, msg):


    recommend_button = QuickReplyButton(
        action=MessageAction(label="สินค้าแนะนำ", text="สินค้าแนะนำ"),
        image_url= 'https://cdn-icons-png.flaticon.com/32/856/856578.png'
    )

    promotion_button = QuickReplyButton(
        action=MessageAction(label="โปรโมชั่น", text="โปรโมชั่น"),
        image_url= 'https://cdn-icons-png.flaticon.com/32/3600/3600488.png'
    )


    cheap_with_quality = QuickReplyButton(
        action=MessageAction(label="สินค้าถูกและดี", text="ถูกและดี"),
        image_url= 'https://cdn-icons-png.flaticon.com/32/10809/10809834.png'
    )
    # test = QuickReplyButton(
    #     action=MessageAction(label="สินค้าถูกและดี", text=styled_text_reply()),
    #     image_url= 'https://cdn-icons-png.flaticon.com/32/10809/10809834.png'
    # )



    quick_reply = QuickReply(
        items=[
            recommend_button,
            promotion_button,
            cheap_with_quality,
            # test

        ]
    )

    return quick_reply

def quick_reply_products(line_bot_api, tk, user_id, msg):


    recommend_button = QuickReplyButton(
        action=MessageAction(label="สินค้าแนะนำ", text="สินค้าแนะนำ"),
        image_url= 'https://cdn-icons-png.flaticon.com/32/856/856578.png'
    )

    promotion_button = QuickReplyButton(
        action=MessageAction(label="โปรโมชั่น", text="โปรโมชั่น"),
        image_url= 'https://cdn-icons-png.flaticon.com/32/3600/3600488.png'
    )
    

    cheap_with_quality = QuickReplyButton(
        action=MessageAction(label="สินค้าถูกและดี", text="ถูกและดี"),
        image_url= 'https://cdn-icons-png.flaticon.com/32/10809/10809834.png'
    )
    

    female_button = QuickReplyButton(
        action=MessageAction(label="สำหรับผู้หญิง", text="ผู้หญิง"),
        image_url= 'https://cdn-icons-png.flaticon.com/32/17390/17390906.png'
    )

    male_button = QuickReplyButton(
        action=MessageAction(label="สำหรับผู้ชาย", text="ผู้ชาย"),
        image_url='https://cdn-icons-png.flaticon.com/32/15735/15735374.png'
    )

    quick_reply = QuickReply(
        items=[
            recommend_button,
            promotion_button,
            cheap_with_quality,
            female_button,
            male_button
        ]
    )

    return quick_reply

def quick_reply_require_detail(line_bot_api, tk, user_id, msg):


    maipheng_button = QuickReplyButton(
        action=MessageAction(label="ไม่แพง", text="ไม่แพง")
    )

    hasib_button = QuickReplyButton(
        action=MessageAction(label="50 ml", text="50 ml")
    )

    twopan_button = QuickReplyButton(
        action=MessageAction(label="ราคาไม่เกิน 2000 บาท", text="2000 บาท")
    )

    summit_button = QuickReplyButton(
        action=MessageAction(label="ยืนยันความต้องการ", text="ยืนยัน"),
        image_url= 'https://cdn-icons-png.flaticon.com/32/12901/12901779.png'
    )

    quick_reply = QuickReply(
        items=[
            maipheng_button,
            hasib_button,
            twopan_button,
            summit_button
        ]
    )

    return quick_reply


def quick_reply_detail(line_bot_api, tk, user_id, msg):


    detail_button = QuickReplyButton(
        action=MessageAction(label="รายละเอียดของสินค้า", text="รายละเอียด"),
        image_url= 'https://cdn-icons-png.flaticon.com/32/11034/11034846.png'
    )
    

    review_button = QuickReplyButton(
        action=MessageAction(label="รีวิว", text="รีวิว"),
        image_url= 'https://cdn-icons-png.flaticon.com/32/10340/10340073.png'
    )

    link_button = QuickReplyButton(
        action=MessageAction(label="ลิงค์สินค้า", text="ลิงค์"),
        image_url= 'https://cdn-icons-png.flaticon.com/32/2885/2885430.png'
    )

    quick_reply = QuickReply(
        items=[
            detail_button,
            review_button,
            link_button
        ]
    )

    return quick_reply

def quick_reply_scent(line_bot_api, tk, user_id, msg):


    fresh_button = QuickReplyButton(
        action=MessageAction(label="สดชื่น", text="สดชื่น"),
        image_url= 'https://cdn-icons-png.flaticon.com/32/6724/6724521.png'
    )

    apple_button = QuickReplyButton(
        action=MessageAction(label="แอปเปิ้ล", text="แอปเปิ้ล"),
        image_url= 'https://cdn-icons-png.flaticon.com/32/3651/3651414.png'
    )

    sport_button = QuickReplyButton(
        action=MessageAction(label="สปอร์ต", text="สปอร์ต"),
        image_url= 'https://cdn-icons-png.flaticon.com/32/2946/2946307.png'
    )

    gentle_button = QuickReplyButton(
        action=MessageAction(label="อ่อนโยน", text="อ่อนโยน"),
        image_url= 'https://cdn-icons-png.flaticon.com/32/3006/3006152.png'
    )

    quick_reply = QuickReply(
        items=[
            fresh_button,
            apple_button,
            sport_button,
            gentle_button
        ]
    )

    return quick_reply

def get_chat_history(uid):
    query = """MATCH (u:User {uid: $uid})-[:ASKED]->(n:User_QA) RETURN n.name as question, n.msg_reply as reply """
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            results = session.run(query, uid=uid)
            chat_history = [{"question": record["question"], "reply": record["reply"]} for record in results]
    return chat_history

def return_message(line_bot_api,tk,user_id,msg):
    global asking , summit , filter , filter_list , selected_product_properties 
    question , msg_reply , types = compute_response(msg ,user_id)
    if msg_reply.lower() in ["เมนู", "menu"]:
        quick_reply = quick_reply_menuu(line_bot_api, tk, user_id, msg)
        line_bot_api.reply_message(
            tk, 
            TextSendMessage(
                text="กรุณาเลือกเมนู", 
                quick_reply=quick_reply
            )
        )
        
    elif msg_reply in ['สินค้าแนะนำ' , 'โปรโมชั่น' , 'ถูกและดี' , 'ลิงค์' , 'รายละเอียด' , 'รีวิว']:
        asking = 0
        if msg_reply == 'ลิงค์':
            line_bot_api.reply_message(
                tk, 
                TextSendMessage(text=f"ลิงค์สินค้า :\n {selected_product_properties['detail_link']}")
            )
        elif msg_reply == 'รายละเอียด':
            if len(selected_product_properties['detail_3']) < 130:
                message = f"รายละเอียดสินค้า :\n {selected_product_properties['detail_2'],selected_product_properties['detail_3']}"
            else:
                message = f"รายละเอียดสินค้า :\n {selected_product_properties['detail_3']}"
            messages = character_reply_specific_product(message , user_id)
            history_graph(user_id , messages , msg_reply)
            line_bot_api.reply_message(
                tk, 
                # TextSendMessage(text=f"รายละเอียดสินค้า :\n {selected_product_properties['detail_3']}")
                TextSendMessage(text=message)
            )
        elif msg_reply == 'รีวิว':
            reply = ""
            comments = collect_product_comments(selected_product_properties['detail_link'])
            for comment in comments:
                comment1 = comment['head_comment'].replace(r'\u200' , ' ')
                comment2 = comment['detail_comment'].replace(r'\u200' , ' ')
                if comment1 != '':
                    reply += f'''รีวิว : {comment1}\n'''
                if comment2 != '':
                    reply += f'''รีวิว : {comment2}\n'''

            if reply == '':
                reply = 'ยังไม่มีรีวิวต่อสินค้าครับ'
            line_bot_api.reply_message(
                tk, 
                TextSendMessage(text= reply)
            )                
        elif msg_reply == 'สินค้าแนะนำ':
            nodes = collect_top_rated()
            products = [format_node(node) for node in nodes]
            flex_messages = build_flex_message(products)
            line_bot_api.reply_message(
                tk, 
                FlexSendMessage(
                    alt_text=msg,  # Set the alternative text for notifications
                    contents=flex_messages["contents"]  # Pass the flex message contents
                )
            )
 

        elif msg_reply == 'โปรโมชั่น': 
            nodes = collect_top_discount()
            products = [format_node(node) for node in nodes]
            flex_messages = build_flex_message(products)
            line_bot_api.reply_message(
                tk, 
                FlexSendMessage(
                    alt_text=msg,  # Set the alternative text for notifications
                    contents=flex_messages["contents"]  # Pass the flex message contents
                )
            )

        else:
            nodes = collect_cheap_with_quality()
            products = [format_node(node) for node in nodes]
            flex_messages = build_flex_message(products)
            line_bot_api.reply_message(
                tk, 
                FlexSendMessage(
                    alt_text=msg,  # Set the alternative text for notifications
                    contents=flex_messages["contents"]  # Pass the flex message contents
                )
            )

    elif msg_reply.lower() in ["ล้างประวัติ", "ล้างแชท", "clear chat" , 'clear' , 'ล้าง']:
        # print('oooooooooooooooooooooooooooooooo')
        asking = 0
        filter = []
        products = []
        filter_list = ['', '', '', '']
        try:
            with GraphDatabase.driver(URI, auth=AUTH) as driver:
                driver.verify_connectivity()
                with driver.session() as session:
                    session.run("""
                        MATCH (u:User {uid: $uid})-[r:ASKED]->(n:User_QA)
                        DETACH DELETE n
                    """, parameters={"uid": user_id})

            line_bot_api.reply_message(tk, TextSendMessage(text='ล้างประวัติ Graph Database เสร็จสิ้น'))
        except Exception as e:
            print(f"Error clearing history: {e}")
            line_bot_api.reply_message(tk, TextSendMessage(text='เกิดข้อผิดพลาดในการล้างประวัติ'))
    
    else:
    #   history_graph(uid = user_id , question= question , answer= msg_reply) 
      if types == "Greeting" and asking == 0:
        print(asking)
        # message = character_reply_requirement(msg_reply , user_id)
        message2 = ((character_reply_requirement('คุณต้องการน้ำหอมสำหรับผู้ชายหรือผู้หญิงครับ' , user_id)).replace('สวัสดีค่ะ' , '')).replace('ค่ะ' , '')
        messages = [
            # TextSendMessage(text=msg_reply),  # First message
            # TextSendMessage(text='คุณต้องการน้ำหอมสำหรับคุณผู้ชาย หรือ คุณผู้หญิงครับ?'),  # Second message
            TextSendMessage(text=msg_reply),  # First message
            TextSendMessage(text=message2),  # Second message
            TextSendMessage(text = '👇🏻',quick_reply=quick_reply_products(line_bot_api, tk, user_id, msg))
        ]
        history_graph(uid = user_id , question= question , answer= msg_reply) 
        line_bot_api.reply_message(tk, messages)
        asking = 1 
      
      else :
          if asking == 1 and types == 'Requirement':
            message = character_reply_requirement('คุณต้องการกลิ่นแบบไหนครับ?' , user_id)
            messages = [TextSendMessage(text=message) , TextSendMessage(text = '👇🏻',quick_reply=quick_reply_scent(line_bot_api, tk, user_id, msg))]
            line_bot_api.reply_message(tk, messages)
            asking = 2
          elif asking == 2 and types == 'Requirement': 
            message = character_reply_requirement('ต้องการอะไรเพิ่มเติมอีกไหมครับ?' , user_id)
            messages = [TextSendMessage(text=message), TextSendMessage(text = '👇🏻',quick_reply=quick_reply_require_detail(line_bot_api, tk, user_id, msg))]
            line_bot_api.reply_message(tk, messages)
            asking = 3
          else:
            if asking == 3 and types == 'Requirement':
                # history_graph(user_id , 'ต้องการอะไรเพิ่มเติมอีกไหมครับ?' , msg)
                if summit == True : 
                    history_graph(user_id , 'ยืนยันความต้องการนะครับ' , msg)
                    asking = 0
                    print(filter)
                    print(requirement(filter))
                    filter_list = requirement(filter)
                    nodes = recommendation(filter_list)
                    products = []
                    if len(nodes) != 0:
                        products = [format_node(node) for node in nodes]
                        flex_messages = build_flex_message(products)
                        line_bot_api.reply_message(
                            tk, 
                            FlexSendMessage(
                                alt_text=msg,  # Set the alternative text for notifications
                                contents=flex_messages["contents"]  # Pass the flex message contents
                            )
                        )
                        # line_bot_api.reply_message(tk , flex_messages)
                        filter = []
                        products = []
                        filter_list = ['', '', '', '']
                        summit = False
                    else :
                        message = character_reply_requirement('ขออภัยผมไม่เจอสินค้าที่เหมาะกับความต้องการของคุณครับ TT' , user_id)
                        messages = [TextSendMessage(text=message)]
                        line_bot_api.reply_message(tk, messages) 
                        filter = []
                        products = []
                        filter_list = ['', '', '', '']
                        summit = False
                    
                else :
                    history_graph(user_id , 'ลักษณะของน้ำหอมที่คุณต้องการ' , msg)
                    message = character_reply_requirement('ต้องการอะไรเพิ่มเติมอีกไหมครับ?' , user_id)
                    messages = [TextSendMessage(text= message), TextSendMessage(text = '👇🏻',quick_reply=quick_reply_require_detail(line_bot_api, tk, user_id, msg))]
                    line_bot_api.reply_message(tk, messages)
            else:
                history_graph(uid = user_id , question= question , answer= msg_reply) 
                line_bot_api.reply_message( tk, TextSendMessage(text=msg_reply) )
            
      
    
      
    

def history_graph(uid, question, answer):
    query = """
    MERGE (u:User {uid: $uid})                   // Create or match User by uid
    MERGE (q:User_QA {uid: $uid, name: $question, msg_reply: $answer}) // Create or match Question uniquely per user
    MERGE (u)-[:ASKED]->(q)                      // Create ASKED relationship between User and Question
    """
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            session.run(query, uid=uid, question=question, answer=answer)
    driver.close()


def requirement(requirements):
    filter_list = ['', '', '', '']
    filters = {'เพศ': 0, 'กลิ่น': 1, 'ปริมาณ': 2}
    
    for require in requirements:
        key, value = require[0], require[1]
        if key in filters:
            filter_list[filters[key]] = value
        else:
            filter_list[3] = value  # Assume the last one is 'else'
    
    return filter_list


def recommendation(filter_list):
    
    # Start building the query string
    query = "MATCH (p:Product) WHERE "
    conditions = []
    params = {}
    
    # Add filters only if values in filter_list are not empty
    if filter_list[0]:  # Gender (เพศ)
        conditions.append("(p.detail CONTAINS $gender OR p.detail_2 CONTAINS $gender OR p.detail_3 CONTAINS $gender)")
        params['gender'] = filter_list[0]
    if filter_list[1]:  # Scent (กลิ่น)
        conditions.append("(p.detail CONTAINS $scent OR p.detail_2 CONTAINS $scent OR p.detail_3 CONTAINS $scent)")
        params['scent'] = filter_list[1]
    if filter_list[2]:  # Quantity (ปริมาณ)
        conditions.append("(p.detail CONTAINS $quantity OR p.detail_3 CONTAINS $quantity)")
        params['quantity'] = filter_list[2]
    if filter_list[3]:  # Price (ราคา <= filter_list[3])
        conditions.append("toFloat(replace(replace(replace(p.price, '฿', ''), ',', ''), ' ', '')) <= toFloat($price)")
        params['price'] = filter_list[3].replace(',', '')  # Clean the input price
    print(conditions)
    # Combine the conditions with AND if there are any
    if conditions:
        query += " AND ".join(conditions)
    else:
        query = "MATCH (p:Product)"  # If no filters, match all products
    
    query += " RETURN p ORDER BY p.rating DESC, p.total_comments DESC LIMIT 3"
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            results = session.run(query , **params)
            nodes = [record["p"] for record in results]
 
    driver.close()
    
    return nodes

def format_node(node):
    properties = dict(node)  # Convert the node to a dictionary if necessary

    formatted_node = {
        "element_id": node.element_id,  # Use node.element_id for the unique identifier
        "id": properties.get("id", ""),
        "name": properties.get("name", ""),
        "detail": properties.get("detail", ""),
        "detail_2": properties.get("detail_2", ""),
        "detail_3": properties.get("detail_3", ""),
        "image": properties.get("image", ""),
        "detail_link": properties.get("detail_link", ""),
        "price": properties.get("price", "").replace('฿', '').replace(',', '').strip(),  # Clean price
        "rating": properties.get("rating", "").replace('rating (เต็ม 5 คะแนน): ', '').strip(),  # Clean rating
        "total_comments": int(properties.get("total_comments", 0)),  # Convert to integer
        "discount": properties.get("discount", 0),
        "origin_price": properties.get("origin_price", "")
    }
    
    return formatted_node

def handle_promotion_text(text):
    if isinstance(text, int) and text == 0:
        return 'No Promotion'
    
    elif isinstance(text, str):
        return text
    
    return text

def adjust_image_url(image_url):
    if '$JPEG$' in image_url:
        return image_url.replace('$JPEG$', '?quality=85&preferwebp=true')
    return image_url

def build_flex_message(products):
    global selected_product_properties 
    flex_bubbles = []
    for product in products:
        bubble = {
                    "type": "bubble", 
                    "hero": {
                        "type": "image",
                        "url": product["image"], 
                        "size": "full",
                        "aspectRatio": "199:265",
                        "aspectMode": "cover"
                    },
                    "body": {
                        "type": "box",
                        "layout": "vertical",
                        "align": "center",
                        "contents": [
                            {
                                "type": "text",
                                "text": product["name"],
                                "weight": "bold",
                                "size": "xl",
                                "color": "#2E2E20",
                                "align": "center",
                                "wrap": True
                            },
                            {
                                "type": "box",
                                "layout": "baseline",
                                "align": "center",
                                "contents": [
                                    {
                                        "type": "icon",
                                        "url": "https://scdn.line-apps.com/n/channel_devcenter/img/fx/review_gold_star_28.png",
                                        "size": "sm"  
                                    },
                                    {
                                        "type": "text",
                                        "text": product["rating"] + ' (' + str(product["total_comments"]) + ')',
                                        "size": "sm",
                                        "color": "#999999",
                                        "margin": "md", 
                                        "flex": 0
                                    }
                                ]
                            },
                            {
                                "type": "box",
                                "layout": "vertical",
                                "contents": [

                                    # Final price
                                    {
                                        "type": "text",
                                        "text": f"฿{product['price']}",
                                        "weight": "bold",
                                        "size": "lg",
                                        "color": "#000000",
                                        "align": "center"
                                        # "decoration": "line-through"
                                    },
                                                        {
                                        "type": "text",
                                        "text": f"{handle_promotion_text(product['discount'])}",
                                        "weight": "bold",
                                        "size": "md",
                                        "color": "#FF6961",
                                        "align": "center"
                                    },
                                    {
                                "type": "button",
                                "style": "primary",
                                "height": "sm",
                                "action": {
                                    "type": "postback",
                                    "label": "Select Product",
                                    "data": f"select_product,{product['detail_link']}",  # Include product ID
                                    "displayText": f"สินค้า: {product['name']}"
                                        },
                                "color": '#2E2E30'  
                                            
                                }                       
                                ]
                            }
                        ]
                    },

                }

        # Add this bubble to the flex_bubbles list
        flex_bubbles.append(bubble)

    # Combine all bubbles into one flex message
    flex_message = {
        "type": "carousel",
        "contents": flex_bubbles
    }

    return {
        "type": "flex",
        "altText": "Product Recommendations",
        "contents": flex_message
    }


def resize_image(input_path, output_path, size=(199, 265)):
    with Image.open(input_path) as img:
        img = img.resize(size, Image.ANTIALIAS)  # Resizing the image
        img.save(output_path)

def collect_top_rated():
    query = """MATCH (p:Product) WITH p, toString(p.total_comments) AS comments RETURN p 
ORDER BY p.rating DESC, toInteger(comments) DESC LIMIT 3"""
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            results = session.run(query)
            nodes = [record["p"] for record in results]

    driver.close()
    
    return nodes

def collect_top_discount():
    query = """MATCH (p:Product)
WHERE p.discount IS NOT NULL AND p.discount CONTAINS '%' 
WITH p, toString(p.discount) AS discountStr
WITH p, discountStr, trim(discountStr) AS trimmedDiscountStr
WITH p, trimmedDiscountStr, split(trimmedDiscountStr, ' ') AS parts
WITH p, trimmedDiscountStr, parts, parts[-1] AS percentageWithSymbol
WITH p, percentageWithSymbol,
     toInteger(replace(percentageWithSymbol, '%', '')) AS discountPercentage
RETURN p
ORDER BY discountPercentage DESC LIMIT 3"""
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            results = session.run(query)
            nodes = [record["p"] for record in results]
    driver.close()
    
    return nodes

def collect_cheap_with_quality():
    query = """MATCH (p:Product)
WHERE p.rating = "rating (เต็ม 5 คะแนน): 5"
WITH p
ORDER BY toInteger(replace(replace(p.price, '฿', ''), ',', '')) ASC, toInteger(p.total_comments) DESC
LIMIT 3
RETURN p"""
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            results = session.run(query)
            nodes = [record["p"] for record in results]
    driver.close()
    return nodes

def collect_selection_product(detail_link):
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
         with driver.session() as session:
            result = session.run(
                '''MATCH (n:Product {detail_link: $detail_link}) RETURN n''', 
                detail_link=detail_link
            )
            record = result.single()  # Fetch a single record
            if record:
                # Extract properties from the node
                node = record["n"]
                product_properties = {
                    "detail": node.get("detail"),
                    "detail_2": node.get("detail_2"),
                    "detail_3": node.get("detail_3"),
                    "detail_link": node.get("detail_link"),
                    "discount": node.get("discount"),
                    "image": node.get("image"),
                    "name": node.get("name"),
                    "origin_price": node.get("origin_price"),
                    "price": node.get("price"),
                    "rating": node.get("rating"),
                    "total_comments": node.get("total_comments"),
                }
                return product_properties
            
def collect_product_comments(detail_link):
        query = """
        MATCH (p {detail_link: $detail_link})-[r:HAS_COMMENT]->(comment)
        RETURN comment
        """
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
         with driver.session() as session:
            result = session.run(query, detail_link=detail_link)
            comments = [record["comment"] for record in result]
            return comments
         
def styled_text_reply():
    flex_message = FlexSendMessage(
        alt_text="Styled Text Message",
        contents=BubbleContainer(
            body=BoxComponent(
                layout="vertical",
                contents=[
                    TextComponent(
                        text="This is bold and large text",
                        weight="bold",   # Makes the text bold
                        size="xl",       # Extra large font size
                        color="#FF0000"  # Red text color
                    ),
                    TextComponent(
                        text="This is normal text",
                        weight="regular", # Normal weight
                        size="md",        # Medium font size
                        color="#0000FF"   # Blue text color
                    ),
                    TextComponent(
                        text="This is small italic text",
                        weight="regular",
                        size="sm",        # Small font size
                        color="#888888",  # Gray text color
                        style="italic"    # Italic text
                    )
                ]
            )
        )
    )

    return flex_message


app = Flask(__name__)
access_token = 'KEQE+K0OdVF7xyxrir+LepnC70OWBNM4DHxcpre9+Nd8eLDZV5c8XMLxticHhyiBX6GAeyM/Z2Y5FQpj1nWQLB53qfhYX5CV6wTRWHuxKhGJuK8lTxzLIrkcja/Q1a4GdOq5wxY7KKehTsBcGmjohgdB04t89/1O/w1cDnyilFU=' 
secret = 'c0dd0a46b60b1233d1bfd0fab9ffbbf3'
line_bot_api = LineBotApi(access_token)              
handler = WebhookHandler(secret)                     
@app.route("/", methods=['POST'])
def linebot():
    global asking , filter_list , summit , filter, selected_product_properties
    body = request.get_data(as_text=True)                    
    try:
        json_data = json.loads(body)                         
        signature = request.headers['X-Line-Signature']      
        handler.handle(body, signature)                      
        msg = json_data['events'][0]['message']['text']      
        tk = json_data['events'][0]['replyToken']
        user_id = json_data['events'][0]['source']['userId'] 
        print(asking)
        if asking == 1 :
            history_graph(user_id , 'ลักษณะของน้ำหอมที่คุณต้องการ' , msg)
        elif asking == 2:
            history_graph(user_id , 'ลักษณะของน้ำหอมที่คุณต้องการ' , msg) 

        return_message(line_bot_api,tk,user_id,msg)
        print(msg, tk) 
    except:
        print(body)                                          
    return 'OK'

@handler.add(PostbackEvent)
def handle_postback(event):
    global selected_product_properties 
    postback_data = event.postback.data
    action, properties = postback_data.split(',')
    if action == 'select_product':
        if properties:
            detail_link = properties
            selected_product_properties = collect_selection_product(detail_link)
            detail_button = QuickReplyButton(
                action=MessageAction(label="รายละเอียดของสินค้า", text="รายละเอียด"),
                image_url= 'https://cdn-icons-png.flaticon.com/32/11034/11034846.png'
            )
            

            review_button = QuickReplyButton(
                action=MessageAction(label="รีวิว", text="รีวิว"),
                image_url= 'https://cdn-icons-png.flaticon.com/32/10340/10340073.png'
            )

            link_button = QuickReplyButton(
                action=MessageAction(label="ลิงค์สินค้า", text="ลิงค์"),
                image_url= 'https://cdn-icons-png.flaticon.com/32/2885/2885430.png'
            )

            quick_reply = QuickReply(
                items=[
                    detail_button,
                    review_button,
                    link_button
                ]
            )

            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="👇🏻" , quick_reply=quick_reply)
            )
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="Sorry, there was an issue with the product selection. Please try again.")
            )

if __name__ == '__main__':
    app.run(port=5000 , debug= True)





app = Flask(__name__)
access_token = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' 
secret = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
line_bot_api = LineBotApi(access_token)              
handler = WebhookHandler(secret)                     
@app.route("/", methods=['POST'])
def linebot():
    global asking , filter_list , summit , filter, selected_product_properties
    body = request.get_data(as_text=True)                    
    try:
        json_data = json.loads(body)                         
        signature = request.headers['X-Line-Signature']      
        handler.handle(body, signature)                      
        msg = json_data['events'][0]['message']['text']      
        tk = json_data['events'][0]['replyToken']
        user_id = json_data['events'][0]['source']['userId'] 
        print(asking)
        if asking == 1 :
            history_graph(user_id , 'ลักษณะของน้ำหอมที่คุณต้องการ' , msg)
        elif asking == 2:
            history_graph(user_id , 'ลักษณะของน้ำหอมที่คุณต้องการ' , msg) 

        print('00000000000000000000000000000000000000')
        return_message(line_bot_api,tk,user_id,msg)
        print(msg, tk) 
    except:
        print(body)                                          
    return 'OK'

