import re

# 파일 읽기
with open('app_incremental.py', 'r', encoding='utf-8') as f:
    content = f.read()

# use_container_width=True를 width='stretch'로 교체
content = content.replace('use_container_width=True', "width='stretch'")

# use_container_width=False를 width='content'로 교체
content = content.replace('use_container_width=False', "width='content'")

# 파일 쓰기
with open('app_incremental.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ 교체 완료!")
print(f"- use_container_width=True → width='stretch'")
print(f"- use_container_width=False → width='content'")
