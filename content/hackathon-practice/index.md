---
emoji: 🔧
title: 개츠비(Gatsby) 접하기
date: '2026-05-06 12:29:00'
author: 지티오
tags: blog gatsby theme 개츠비 테마
categories: 개발
---


## 계기
이틀 뒤에 해커톤에 백엔드 역할로 나갈 예정이다. 사실 잘 모르겠다.. 백엔드 되기를 빌어본다. 암튼 연습을 위해서 claudeAI에게 6시간짜리 프로젝트를 부탁했다. 프론트를 작성하게 시키고 주어진 API 명세서를 바탕으로 내가 백엔드 작업을 하는것이다. 환경설정을 배우면서 하다보니 8시간이 되었지만 구조를 알게되었다.

## 만들려는 것
냉장고를 찍어 올리면 냉장고에 있는 재료들을 분석해서 레시피를 보내는 서비스이다.  


## 백엔드 기술 스택
프레임워크: expressjs  
런타임검증: zod  
ORM: Prisma  
Auth: Passport  
  
db: SQLite  

### src/app.ts

```ts
import express from "express";
import helmet from "helmet";
import cors from "cors";
import { env } from "./config.js";
import "./passport/jwt.strategy.js"
import passport from "passport";
import authRouter from "./routes/auth.route.js";
import analysesRouter from "./routes/analyses.route.js";
import { ErrorHandler } from "./middleware/error.js";

export const app = express()

app.use(helmet())
app.use(cors({origin:env.CORS_ORIGIN, credentials:true}))
app.use(express.json())
app.use(passport.initialize())

app.use("/api/health", (_req, res) => {
    return res.status(200).json({status: "ok"})
})
app.use("/api/auth", authRouter)
app.use("/api/analyses", analysesRouter)

app.use(ErrorHandler)

```

<br/>
<br/>
  
  
```ts
app.use(helmet()) 
```
> 헤더의 보안설정을 위한 미들웨어이다  

<br/>

```ts
app.use(cors({origin:env.CORS_ORIGIN, credentials:true}))
```
> 동일 출처 정책(Same-Origin Policy) 해제  
  
<br/>

```ts
app.use(express.json())
```
> 파싱  
  
<br/>

```ts

app.use(passport.initialize())
```
> passport strategy 적용  
  
### src/server.ts

```ts
import { createServer } from "http";
import { app } from "./app.js";
import { env } from "./config.js";


const httpServer = createServer(app)


httpServer.listen(Number(env.PORT), () => {
    console.log(`Server is running on port ${env.PORT}`)
    console.log(`http://localhost:${env.PORT}`)
})

```
> socket.io 등 확장을 위해 이런식으로 했다.

### src/passport/jwt.strategy.ts
```ts
import passport from "passport";
import { ExtractJwt, Strategy as JwtStrategy } from "passport-jwt";
import { env } from "../config.js";
import { findUserById } from "../service/auth.service.js";


const jwtOptions = {
    jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
    secretOrKey: env.JWT_SECRET,
}


passport.use(new JwtStrategy(jwtOptions, async (payload, done) => {
    try{
        const user = await findUserById(payload.id)
        if (!user) return done(null, false)
        return done(null, user)
    } catch (error) {
        return done(error, false)
    }
    
}))
```
> 얘는 JWT를 실제로 생성해서 보내는 놈이 아니다. 검증만 하는 미들웨어인 셈이다. 중요하다 이거

> 원래는 JWT 미들웨어 구현이 더 쉽지만 확장성 때문에 passport를 선택했다.

<br/>
<br/>

```ts
jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken()
```
> jwt정보를 "헤더"에서 가져오라는 뜻이다 많이 보던 Bearer token 형태가 맞다


```ts
import { Router } from "express";
import { authenticateJWT } from "../middleware/auth.js";
import { validateBody } from "../middleware/validate.js";
import { analyseSchema } from "../types/schemas.js";
import * as controller from "../controller/analyses.controller.js";
import multer from "multer";



const router = Router();

export const uploadFiles = multer();

router.post("", authenticateJWT,uploadFiles.single("image"), validateBody(analyseSchema), controller.analyse)


export default router;
```
> multer는 아무런 인자가 없을땐 메모리에 저장을 한다. 우리는 받은 이미지를 바로 써야하기 때문에 메모리에만 저장을 한다. 만약 다시 꺼내온다거나 그런 전략을 사용하는 경우에는 cloudinary에 저장하는 방향이 있다.

> 순서가 중요하다. multer가 먼저 req.file이랑 req.body를 만들어야됨.