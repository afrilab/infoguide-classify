# Top Discriminative TF-IDF Terms per Class

- N = 94 documents
- Class distribution: Forms_Structured=20, Policy_Procedure_Contract=29, Reports=45
- TF-IDF preprocessing: ngram=[1, 2], stop_words=english, lowercase=True, min_df=1, vocab_size=41316
- Score = mean(TF-IDF | in class) − mean(TF-IDF | other classes); higher means more distinctive to the class.

## Forms_Structured  (N=20)

| Rank | Term | In-class mean | Out-class mean | Score |
| ---: | --- | ---: | ---: | ---: |
| 1 | `borrower` | 0.1525 | 0.0011 | 0.1515 |
| 2 | `reserve` | 0.0813 | 0.0015 | 0.0797 |
| 3 | `participant` | 0.0752 | 0.0002 | 0.0749 |
| 4 | `agreement` | 0.0758 | 0.0015 | 0.0743 |
| 5 | `reserve bank` | 0.0686 | 0.0003 | 0.0682 |
| 6 | `excess` | 0.0594 | 0.0000 | 0.0594 |
| 7 | `title` | 0.0567 | 0.0002 | 0.0564 |
| 8 | `excess balance` | 0.0563 | 0.0000 | 0.0563 |
| 9 | `agent` | 0.0566 | 0.0003 | 0.0563 |
| 10 | `oc` | 0.0539 | 0.0000 | 0.0539 |
| 11 | `oc 10` | 0.0539 | 0.0000 | 0.0539 |
| 12 | `correspondent` | 0.0550 | 0.0036 | 0.0514 |
| 13 | `balance account` | 0.0492 | 0.0000 | 0.0492 |
| 14 | `authorized` | 0.0501 | 0.0013 | 0.0488 |
| 15 | `account` | 0.0484 | 0.0041 | 0.0442 |
| 16 | `balance` | 0.0430 | 0.0025 | 0.0405 |
| 17 | `resolutions` | 0.0400 | 0.0003 | 0.0398 |
| 18 | `operating circular` | 0.0374 | 0.0000 | 0.0374 |
| 19 | `authorization` | 0.0376 | 0.0005 | 0.0372 |
| 20 | `circular` | 0.0364 | 0.0002 | 0.0362 |

## Policy_Procedure_Contract  (N=29)

| Rank | Term | In-class mean | Out-class mean | Score |
| ---: | --- | ---: | ---: | ---: |
| 1 | `fatf` | 0.0944 | 0.0003 | 0.0942 |
| 2 | `risk` | 0.0828 | 0.0122 | 0.0706 |
| 3 | `guidance` | 0.0492 | 0.0024 | 0.0469 |
| 4 | `credit risk` | 0.0377 | 0.0011 | 0.0366 |
| 5 | `credit` | 0.0495 | 0.0130 | 0.0365 |
| 6 | `ccr` | 0.0362 | 0.0004 | 0.0358 |
| 7 | `principles` | 0.0364 | 0.0043 | 0.0322 |
| 8 | `committee` | 0.0342 | 0.0082 | 0.0259 |
| 9 | `operational` | 0.0302 | 0.0047 | 0.0256 |
| 10 | `operational resilience` | 0.0251 | 0.0004 | 0.0247 |
| 11 | `countries` | 0.0272 | 0.0028 | 0.0244 |
| 12 | `supervision` | 0.0345 | 0.0111 | 0.0234 |
| 13 | `cft` | 0.0247 | 0.0018 | 0.0229 |
| 14 | `laundering` | 0.0257 | 0.0033 | 0.0224 |
| 15 | `exposures` | 0.0245 | 0.0022 | 0.0223 |
| 16 | `counterparty` | 0.0241 | 0.0018 | 0.0223 |
| 17 | `financial inclusion` | 0.0244 | 0.0022 | 0.0222 |
| 18 | `aml cft` | 0.0237 | 0.0018 | 0.0220 |
| 19 | `money laundering` | 0.0253 | 0.0033 | 0.0219 |
| 20 | `aml` | 0.0242 | 0.0024 | 0.0218 |

## Reports  (N=45)

| Rank | Term | In-class mean | Out-class mean | Score |
| ---: | --- | ---: | ---: | ---: |
| 1 | `government` | 0.0399 | 0.0008 | 0.0391 |
| 2 | `mr` | 0.0361 | 0.0000 | 0.0361 |
| 3 | `world` | 0.0384 | 0.0027 | 0.0358 |
| 4 | `world bank` | 0.0363 | 0.0017 | 0.0346 |
| 5 | `sector` | 0.0402 | 0.0071 | 0.0331 |
| 6 | `assessment` | 0.0387 | 0.0058 | 0.0329 |
| 7 | `public` | 0.0368 | 0.0053 | 0.0315 |
| 8 | `development` | 0.0327 | 0.0026 | 0.0301 |
| 9 | `pi` | 0.0298 | 0.0000 | 0.0298 |
| 10 | `pfm` | 0.0294 | 0.0000 | 0.0294 |
| 11 | `erusolcsid cilbup` | 0.0299 | 0.0010 | 0.0289 |
| 12 | `cilbup` | 0.0299 | 0.0010 | 0.0289 |
| 13 | `erusolcsid` | 0.0299 | 0.0010 | 0.0289 |
| 14 | `dezirohtua` | 0.0299 | 0.0010 | 0.0289 |
| 15 | `dezirohtua erusolcsid` | 0.0299 | 0.0010 | 0.0289 |
| 16 | `financial` | 0.0602 | 0.0319 | 0.0284 |
| 17 | `team` | 0.0255 | 0.0000 | 0.0255 |
| 18 | `expenditure` | 0.0246 | 0.0000 | 0.0246 |
| 19 | `financial sector` | 0.0271 | 0.0032 | 0.0239 |
| 20 | `budget` | 0.0239 | 0.0008 | 0.0231 |

