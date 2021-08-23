## python基础语法


```python
#1.基础操作
age = 20  		# 声明一个变量age 用来存储一个数字 20
1+1		        # 基础数学加法
print('Hello World!')   # 打印Hello World!

```


```python
#2.条件判断if
if 1 == 2: # 如果 if 跟随的条件为 假 那么不执行属于if 的语句,然后寻找 else
    print("假的")
else: # 寻找到 else 之后 执行属于else中的语句
    print("1==2是假的")
```


```python
#3.循环操作---for
for i in range(5):
    print(i)
```


```python
#3.循环操作---while
sum = 0
n = 99
while n > 0:
    sum = sum + n
    n = n - 1
print(sum)
```


```python
#4.break、continue、pass
#break语句可以跳出 for 和 while 的循环体
n = 1
while n <= 100:
    if n > 10:
        break
    print(n)
    n += 1

```

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10



```python
#continue语句跳过当前循环，直接进行下一轮循环
n = 1
while n < 10:
    n = n + 1
    if n % 2 == 0:
        continue
    print(n)

```

    3
    5
    7
    9



```python
 #pass是空语句，一般用做占位语句，不做任何事情
 for letter in 'Room':
    if letter == 'o':
        pass
        print('pass')
    print(letter)


```

    R
    pass
    o
    pass
    o
    m



```python
#5.数据类型---Number(数字)
#Python支持int, float, complex三种不同的数字类型
a = 3
b = 3.14
c = 3 + 4j
print(type(a), type(b), type(c))
```

    <class 'int'> <class 'float'> <class 'complex'>



```python
#5.数据类型---String（字符串）
#支持字符串拼接、截取等多种运算
a = "Hello"
b = "Python"
print("a + b 输出结果：", a + b)
#print("a[1:4] 输出结果：", a[1:4])
```

    a + b 输出结果： HelloPython



```python
#5.数据类型---List（列表）
#列表是写在方括号 [] 之间、用逗号分隔开的元素列表。
#列表索引值以 0 为开始值，-1 为从末尾的开始位置。
list = ['abcd', 786 , 2.23, 'runoob', 70.2]
print(list[1:3])

#tinylist = [123, 'runoob']
#print(list + tinylist)
```

    [786, 2.23]



```python
#5.数据类型---Tuple（元组）
#tuple与list类似，不同之处在于tuple的元素不能修改。tuple写在小括号里，元素之间用逗号隔开。
#元组的元素不可变，但可以包含可变对象，如list。
t1 = ('abcd', 786 , 2.23, 'runoob', 70.2)
t2 = (1, )
t3 = ('a', 'b', ['A', 'B'])
t3[2][0] = 'X'
print(t3)
```

    ('a', 'b', ['X', 'B'])



```python
#5.数据类型---dict（字典）
#字典是无序的对象集合，使用键-值（key-value）存储，具有极快的查找速度。
#键(key)必须使用不可变类型。
#同一个字典中，键(key)必须是唯一的。
d = {'Michael': 95, 'Bob': 75, 'Tracy': 85}
print(d['Michael'])
```

    95



```python
#5.数据类型---set（集合）
#set和dict类似，也是一组key的集合，但不存储value。由于key不能重复，所以，在set中，没有重复的key。
#set是无序的，重复元素在set中自动被过滤。
s = set([1, 1, 2, 2, 3, 3])
print(s)
```

    {1, 2, 3}

