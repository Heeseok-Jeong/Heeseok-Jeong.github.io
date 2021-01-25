---
layout : post
title : 기타 그래프 알고리즘
subtitle : 코딩 테스트에서 자주 등장하는 기타 그래프 이론들
tags : [Problem Solving]
author : Heeseok Jeong
comments : True
use_math: True
sitemap :
  changefreq : daily
  priority : 1.0
---

# 목차
<br>

- [특징](#특징)
- [Disjoint Sets](#disjoint-sets)
- [신장 트리](#신장-트리)
- [위상 정렬](#위상-정렬)

<br>
<hr>
<br>

# 특징

- DFS/BFS, 최단 경로 알고리즘 등 다양한 알고리즘이 그래프 유형임
- 매우 많은 그래프 유형 중 코딩 테스트 출제 비중은 낮지만 꼭 제대로 알아야 하는 알고리즘들임
- 개념들을 제대로 이해하면 코딩 테스트에서 다양한 응용 문제 해결 가능
- 그래프란?
    - 노드와 간선의 정보를 가진 자료구조
    - 서로 다른 개체가 **연결**돼있다 → 그래프 생각해볼 것
    - 구현 방식
        - 인접 행렬 (2차원 리스트)
            - 공간 복잡도 : O(V^2)
            - 접근 시간 : O(1)
            - 플로이드 워셜에서 사용했음
        - 인접 리스트(리스트 or 딕트) 로 구현
            - 공간 복잡도 : O(E)
            - 접근 시간 : O(V)
            - 다익스트라에서 사용했음
- 그래프와 트리

    ||그래프|트리|
	|---|---|---|
    |방향성|방향 그래프 or 무방향 그래프|방향 그래프|
    |순환성|순환 or 비순환|비순환
    |루트 노드 존재|X|O|
    |노드간 관계성|부모 자식 X|부모 자식 O|
    |모델의 종류|네트워크 모델|계층 모델|

<br>

<hr>

<br>

# Disjoint Sets

- 공통 원소가 없는 두 집합, {1, 2} ↔ {3, 4}
- 서로소 집합 자료구조 (a.k.a union-find 자료구조)
    - 서로소 부분 집합들로 나누어진 원소들의 데이터를 처리하기 위한 자료구조
    - union, find 연산으로 조작
        - union
            - 원소가 포함된 두 집합을 하나로 합치는 연산
            - 간선으로 취급
        - find
            - 특정한 원소가 속한 집합이 어떤 집합인지 알려주는 연산
    - 트리 구조로 집합을 표현함
        - union
            - 두 노드 A, B 에 대해 유니언을 수행하면, 각각의 루트 노드를 재귀적으로 찾고 해당 집합의 모든 원소에 대해 A 의 루트 노드를 부모노드로 설정한다.
- 기본 방법은 find, 부모 노드를 찾을 때 테이블을 모두 살펴보기 때문에 비효율적, O(VM) (V : 노드 개수, M : 유니언 연산 횟수)
- 경로 압축 기법을 사용하면 시간 복잡도 개선 가능!
    - 부모 노드를 찾으러 재귀적으로 들어갔다가 나올 때 모두 부모 노드 갱신
    - 마지막에 모든 원소에 대해 find 수행

```python
# 특정 원소가 속한 집합을 찾기
def find_parent(parent, x):
	# 루트 노드가 아니라면, 루트 노드를 찾을 때까지 재귀 호출
	if parent[x] != x:
		parent[x] = find_parent(parent, parent[x])
	return x

# 두 원소가 속한 집합을 합치기
def union_parent(parent, a, b):
	a = find_parent(parent, a)
	b = find_parent(parent, b)
	if a < b:
		parent[b] = a
	else:
		parent[a] = b

# 노드의 개수와 간선(union 연산) 의 개수 입력받기
v, e = map(int, input().split())
parent = [0] * (v+1)

# 부모 테이블상에서, 부모를 자기 자신으로 초기화
for i in range(1, v+1):
	parent[i] = i

# union 연산을 각각 수행
for i in range(e):
	a, b = map(int, input().split())
	union_parent(parent, a, b)

# 각 원소가 속한 집합 출력
for i in range(1, v+1):
	print(find_parent(panret, i), end = ' ')
	
# 부모 테이블 내용 출력
for i in range(1, v+1):
	print(parent[i], end = ' ')

```

- 시간 복잡도 : O(V + M(1 + $log_\mathit{2-M/V}V$))
    - 노드 개수 V, 최대 V-1 개의 union 연산, M 개의 find 연산
- 시간 복잡도를 줄일 다른 방법들도 있지만 코테용으로는 경로 압축 충분

### 서로소 집합을 활용한 사이클 판별

- 서로소 집함으로 **무방향 그래프** 내에서 사이클 판별 가능 (방향 그래프에서는 DFS 로 판별 가능)
- union 연산을 수행하며 같은 부모 노드를 지닌 노드들을 찾음 → 사이클

```python
cycle = False # 사이클 발생 여부

for i in range(e):
	a, b = map(int, input().split())
	# 사이클이 발생한 경우 종료
	if find_parent(parent, a) == find_parent(parent, b):
		cycle = True
		break
	# 아니라면 union(합집합) 수행
	else:
		union_parent(parent, a, b)

if cycle:
	print("Cycle O")
else:
	print("Cycle X")
```

<br>

<hr>

<br>

# 신장 트리

- 하나의 그래프가 있을 때 모든 노드를 포함하면서 사이클이 존재하지 않는 부분 그래프 → 트리 성립 조건과 같음 (신장 트리라고 불리는 이유)

### 크루스칼 알고리즘

- 최소한의 비용으로 신장 트리 찾기 (MST, Minumum Spanning Tree)
- 그리디 알고리즘
- 모든 간선에 대해 정렬 (비용순) 을 수행한 후, 가장 거리가 짧은 간선부터 집합에 포함시킴 (사이클이 발생할 경우 집합에 넣지 않음)
- 최종 간선의 개수 = 노드 개수 - 1 (트리 조건과 같음)
- 집합에 넣는다 → 두 노드에 대해 union 수행 (중요한 개념, 직접 집합 만드는거 아님)
- 두 노드의 부모 노드 먼저 확인하고, 같지 않은 경우 집합에 넣음!
- 최소 신장 트리에 포함된 간선의 비용 모두 더하면 최종 비용이 됨

```python
# 특정 원소가 속한 집합을 찾기
def find_parent(parent, x):
	if parent[x] != x:
		parent[x] = find_parent(parent, parent[x])
	return x

# 두 원소가 속한 집합을 합치기
def union_parent(parent, a, b):
	parent_a = find_parent(parent, a)
	parent_b = find_parent(parent, b)
	
	if p_a < p_b:
		parent[p_b] = p_a
	else:
		parent[p_a] = p_b

# 노드의 개수와 간선의 개수 입력받기
v, e = map(int, input().split())
parent = [0] * (v+1)

# 모든 간선을 담을 리스트와 최종 비용을 담을 변수
edges = []
result = 0

# 부모 테이블상에서, 부모를 자기 자신으로 초기화
for i in range(1, v+1):
	parent[i] = i

# 모든 간선에 대한 정보를 입력받기
for _ in range(e):
	a, b, cost = map(int, input().split())
	# 비용순으로 정렬하기 위해서 튜플의 첫번째 원소를 비용으로 설정
	edges.append((cost, a, b))

# 간선을 비용순으로 정렬
edges.sort()

# 간선을 하나씩 확인하며
for edge in edges:
	cost, a, b = edge
	# 사이클이 발생하지 않는 경우에만 집합에 포함
	if find_parent(a) != find_parent(b):
		union_parent(parent, a, b)
		result += cost

print(result)
```

- 시간 복잡도 : O(ElogE) (간선 정렬시간이 제일 큼)

<br>
<hr>

<br>

# 위상 정렬

- 순서가 정해져 있는 일련의 작업을 차례대로 수행해야 할 때 사용할 수 있는 알고리즘
- 방향 그래프의 모든 노드를 방향성에 거스르지 않도록 순서대로 나열하는 것
- 진입차수, Indegree
    - 특정한 노드로 들어오는 간선의 개수
- 과정
    1. 진입차수가 0 인 노드를 큐에 넣는다. → 들어오는게 없으니 시작점이 됨
    2. 큐가 빌 때까지 다음의 과정 반복
        1. 큐에서 원소를 꺼내 해당 노드에서 출발하는 간선을 그래프에서 제거
        2. 새롭게 진입차수가 0 이 된 노드를 큐에 넣음
- 모든 원소를 방문하기 전에 큐가 비면 (큐에서 V 번 원소 꺼내기전에 비면) 사이클 존재 (사이클인 원소들은 큐에 못들어가기 때문),  보통 사이클 없다고 명시되는 경우가 많음, 사이클 있을 때 처리법이 따로 있음
- 답 (순서) 여러개 존재 가능
- 연결 리스트 사용

```python
from collections import deque

# 노드의 개수와 간선의 개수 입력받기
v, e = map(int, input().split())
# 모든 노드에 대한 진입차수는 0 으로 초기화
indegree = [0] * (v+1)
# 각 노드에 연결된 간선 정보를 담기 위한 연결 리스트 초기화
graph = [[] for _ in range(v+1)]

# 방향 그래프의 모든 간선 정보 입력받기
for _ in range(e):
	a, b = map(int, input().split())
	graph[a].append(b) # 노드 a 에서 b 로 이동 가능
	# 진입차수 1 증가
	indegree[b] += 1

# 위상 정렬 함수
def topology_sort():
	result = [] # 알고리즘 수행 결과를 담을 리스트
	q  = deque()

	# 처음 시작할 때는 진입차수가 0 인 노드 큐에 삽입
	for i in range(1, v+1):
		if indegree[i] == 0:
			q.append(i)

		# 큐가 빌 때까지 반복
		while q:
			now = q.popleft()
			result.append(now)
			# 해당 원소와 연결된 노드들의 진입차수에서 1 빼기
			for x in graph[now]:
				indegree[x] -= 1
				# 진입차수 0 인 노드 큐에 삽입
				if indegree[x] == 0:
					q.append(x)

topology_sort()

# 위상 정렬을 수행한 결과 출력
for i in result:
	print(i, end = ' ')
```

- 시간 복잡도 : O(V + E) (모든 노드에 대해 각 노드의 간선 확인)

<br>
<hr>
<br>

# 참조
<br>

- part2, 이것이 취업을 위한 코딩테스트다 with 파이썬
