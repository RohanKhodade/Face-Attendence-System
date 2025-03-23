import tkinter as tk
from newuser import register_new_user
from detectface import recognize_face 

root = tk.Tk()
root.title("Face Recognition System")
root.geometry("400x300")

def markattendence():
    user_name,similarity=recognize_face()
    result_label.config(text="Face Recognized: {}\nConfidence: {:.2f}".format(user_name, similarity))

tk.Label(root, text="Face Recognition System", font=("Arial", 16)).pack(pady=20)

register_button=tk.Button(root,text="Register",command=register_new_user,width=20,height=2)
register_button.pack(pady=2)

mark_attendence=tk.Button(root,text="Mark Attendence",command=markattendence,width=20,height=2)
mark_attendence.pack(pady=2)

result_label=tk.Label(root,text="",font=("Arial",12))
result_label.pack(pady=10)

exit_button=tk.Button(root,text="Exit",command=root.quit,width=20,height=2)
exit_button.pack(pady=2)
root.mainloop()

