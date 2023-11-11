css = '''
<style>
.chat-message {
    min-height: 1rem; padding: 1.25rem; border-radius: 0.25rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 10%;
}
.chat-message .avatar img {
  max-width: 50px;
  max-height: 50px;
  border-radius: 3px;
  object-fit: cover;
}
.chat-message .message {
  width: 90%;
  padding: 0 0.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://easydrawingguides.com/wp-content/uploads/2022/09/Easy-robot-face-11.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://img.myloview.com/stickers/cartoon-face-isolated-vector-icon-wide-delighted-smile-facial-emoji-happy-funny-creature-happy-human-emotion-comic-face-with-toothy-smiling-mouth-and-round-eyes-700-248728891.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''