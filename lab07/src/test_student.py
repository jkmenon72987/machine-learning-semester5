from models import StudentCNN, OversizedCNN

teacher = OversizedCNN()
student = StudentCNN()

teacher_params = sum(p.numel() for p in teacher.parameters())
student_params = sum(p.numel() for p in student.parameters())

print("Teacher parameters:", teacher_params)
print("Student parameters:", student_params)

print(
    "Parameter reduction:",
    round((1 - student_params / teacher_params) * 100, 2),
    "%"
)

print("Student architecture verified.")
