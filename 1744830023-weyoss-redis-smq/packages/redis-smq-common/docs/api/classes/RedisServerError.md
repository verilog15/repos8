[RedisSMQ Common Library](../../../README.md) / [Docs](../../README.md) / [API Reference](../README.md) / RedisServerError

# Class: RedisServerError

## Hierarchy

- [`RedisSMQError`](RedisSMQError.md)

  ↳ **`RedisServerError`**

  ↳↳ [`RedisServerBinaryNotFoundError`](RedisServerBinaryNotFoundError.md)

  ↳↳ [`RedisServerUnsupportedPlatformError`](RedisServerUnsupportedPlatformError.md)

## Table of contents

### Constructors

- [constructor](RedisServerError.md#constructor)

### Properties

- [cause](RedisServerError.md#cause)
- [message](RedisServerError.md#message)
- [stack](RedisServerError.md#stack)
- [prepareStackTrace](RedisServerError.md#preparestacktrace)
- [stackTraceLimit](RedisServerError.md#stacktracelimit)

### Accessors

- [name](RedisServerError.md#name)

### Methods

- [captureStackTrace](RedisServerError.md#capturestacktrace)

## Constructors

### constructor

• **new RedisServerError**(`message?`): [`RedisServerError`](RedisServerError.md)

#### Parameters

| Name | Type |
| :------ | :------ |
| `message?` | `string` |

#### Returns

[`RedisServerError`](RedisServerError.md)

#### Inherited from

[RedisSMQError](RedisSMQError.md).[constructor](RedisSMQError.md#constructor)

## Properties

### cause

• `Optional` **cause**: `unknown`

#### Inherited from

[RedisSMQError](RedisSMQError.md).[cause](RedisSMQError.md#cause)

___

### message

• **message**: `string`

#### Inherited from

[RedisSMQError](RedisSMQError.md).[message](RedisSMQError.md#message)

___

### stack

• `Optional` **stack**: `string`

#### Inherited from

[RedisSMQError](RedisSMQError.md).[stack](RedisSMQError.md#stack)

___

### prepareStackTrace

▪ `Static` `Optional` **prepareStackTrace**: (`err`: `Error`, `stackTraces`: `CallSite`[]) => `any`

Optional override for formatting stack traces

**`See`**

https://v8.dev/docs/stack-trace-api#customizing-stack-traces

#### Type declaration

▸ (`err`, `stackTraces`): `any`

##### Parameters

| Name | Type |
| :------ | :------ |
| `err` | `Error` |
| `stackTraces` | `CallSite`[] |

##### Returns

`any`

#### Inherited from

[RedisSMQError](RedisSMQError.md).[prepareStackTrace](RedisSMQError.md#preparestacktrace)

___

### stackTraceLimit

▪ `Static` **stackTraceLimit**: `number`

#### Inherited from

[RedisSMQError](RedisSMQError.md).[stackTraceLimit](RedisSMQError.md#stacktracelimit)

## Accessors

### name

• `get` **name**(): `string`

#### Returns

`string`

#### Inherited from

RedisSMQError.name

## Methods

### captureStackTrace

▸ **captureStackTrace**(`targetObject`, `constructorOpt?`): `void`

Create .stack property on a target object

#### Parameters

| Name | Type |
| :------ | :------ |
| `targetObject` | `object` |
| `constructorOpt?` | `Function` |

#### Returns

`void`

#### Inherited from

[RedisSMQError](RedisSMQError.md).[captureStackTrace](RedisSMQError.md#capturestacktrace)
