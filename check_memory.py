import subprocess
import sys
# это файл для проверки работы кода для практики 5, код в нем я перезаписывал и менял по ходу различных вариантов проверок

# дочерний процесс записывает 5 реплик
subprocess.run([sys.executable, "-c", """
import memory
memory.create_schema()
memory.clear("t2")
for i in range(1, 6):
    memory.save("t2", "user", f"реплика {i}")
print("записано 5 реплик")
"""], check=True)

# другой процесс читает — история должна быть на месте
subprocess.run([sys.executable, "-c", """
import memory   
w = memory.window("t2", 10)
print([m.content for m in w])
assert len(w) == 5, "история не пережила перезапуск"
assert [m.content for m in w] == [f"реплика {i}" for i in range(1, 6)]
print("перезапуск: история на месте - ок")
"""], check=True)


# изоляция сессий
import memory
memory.clear("iso_a")
memory.clear("iso_b")
memory.save("iso_a", "user", "реплика сессии A")
memory.save("iso_b", "user", "реплика сессии B")

w_a = memory.window("iso_a", 10)
w_b = memory.window("iso_b", 10)

assert len(w_a) == 1 and w_a[0].content == "реплика сессии A", "Сессия A видит лишнее"
assert len(w_b) == 1 and w_b[0].content == "реплика сессии B", "Сессия B видит лишнее"
print("изоляция сессий: каждая видит только свои реплики - ок")

memory.clear("iso_a")
memory.clear("iso_b")