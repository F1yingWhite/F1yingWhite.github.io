---
title: Hooks
description: ""
image: ""
published: 2025-01-01
tags:
  - React
  - 前端
category: React
draft: false
---

# useContext

`useContext` 是一个 React Hook,可以让你读取订阅中的 context

```tsx
const value = useContext(someContext)
```

## 用法

### 像组件树深层传递数据

```tsx
function myPage(){
	return (
	<ThemeContext.Provider value="dark">
      <Form />
    </ThemeContext.Provider>
	)
}

function Button(){
	const theme = useContext(ThemeContext);
	...
}
```

## 数据传递

```tsx
function MyPage() {
  const [theme, setTheme] = useState('dark');
  return (
    <ThemeContext.Provider value={theme}>
      <Form />
      <Button onClick={() => {
        setTheme('light');
      }}>
       Switch to light theme
      </Button>
    </ThemeContext.Provider>
  );
}
```

# useMemo

`useMemo` 是用来在每次重新渲染的时候缓存计算结果.

:::warning

这只是用来优化性能的手段,少了他代码代码也需要正常运行

:::

```tsx
const cachedValue = useMemo(calucatedValue,dependencies)
```

用法

- 跳过代价高昂的重新计算
- 跳过组件的重新渲染
- 防止频繁触发 Effect
参数
- `calculateValue`：要缓存计算值的函数。它应该是一个没有任何参数的纯函数，并且可以返回任意类型。React 将会在首次渲染时调用该函数；在之后的渲染中，如果 `dependencies` 没有发生变化，React 将直接返回相同值。否则，将会再次调用 `calculateValue` 并返回最新结果，然后缓存该结果以便下次重复使用。
- `dependencies`：所有在 `calculateValue` 函数中使用的响应式变量组成的数组。响应式变量包括 props、state 和所有你直接在组件中定义的变量和函数。
返回值
在初次渲染时，`useMemo` 返回不带参数调用 `calculateValue` 的结果。
在接下来的渲染中，如果依赖项没有发生改变，它将返回上次缓存的值；否则将再次调用 `calculateValue`，并返回最新结果。

## 跳过组件的重新渲染

默认情况下,当一个组件被重新渲染的时候,react 会递归的重新渲染他的所有子组件.但是如果你已经确认重新渲染很慢，你可以通过将它包装在 [`memo`](https://zh-hans.react.dev/reference/react/memo) 中，这样当它的 props 跟上一次渲染相同的时候它就会跳过本次渲染：

```tsx
import {memo} from 'react';

const List = memo(function List({items})){
	...
}
```

这里如果 List 的所有 props 都和之前一致,那么他将跳过重新渲染,这就是缓存计算重要的地方

# useRef

他能帮忙引用一个不需要渲染的值

```tsx
import {useRef} from 'react';

function MyComponent(){
	const intervalRef = useRef(0);
}
```

initvalue:ref 对象的初始值,可以使任意类型的值,这个参数在首次渲染后被忽略,通过 ref.current 获取当前的信息

**改变 ref 不会触发重新渲染**。这意味着 ref 是存储一些不影响组件视图输出信息的完美选择。

## 通过 Ref 操作 Dom

```tsx
import { useRef } from 'react';

export default function Form() {
  const inputRef = useRef(null);

  function handleClick() {
    inputRef.current.focus();
  }

  return (
    <>
      <input ref={inputRef} />
      <button onClick={handleClick}>
        聚焦输入框
      </button>
    </>
  );
}
```

React 将会把 DOM 节点设置为 ref 对象的 `current` 属性。现在可以借助 ref 对象访问 `<input>` 的 DOM 节点，并且可以调用类似于 [`focus()`](https://developer.mozilla.org/zh-CN/docs/Web/API/HTMLElement/focus) 的方法：

```tsx
function handleClick() { 
	inputRef.current.focus();
}
```

当节点从屏幕上移除时，React 将把 `current` 属性设置回 `null`

