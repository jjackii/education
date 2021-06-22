use adventureworks;

/** sql 순서
select 
from table, view (1)
where (filtering) (2)
group by (3)
having (4)
order by (final) (5)
**/

## 1

select * -- businessEntityid
-- , employeeid
-- , contactid
-- , Loginid
from employee
limit 0,10;

-- disk->memory->cpu


# distinct (중복제거&grouping)
select title from employee;

-- 역할의 개수/종류를 알고 싶다면 
select distinct title
from employee;

-- title과 gender가 고유한 레코드(튜플) 조회
select distinct title, gender
from employee;


# where (조건절)
-- (문자형) 
select * from employee 
-- where title = 'tool designer'
where title like 'tool designer'
where title like '%tool designer' -- tool designer로 끝나는 거 찾기
where title like 'designer%'	  -- designer 로 시작하는 거 찾기
where title like '%designer%'	  -- 중간에 designer 가 있는 거 찾기


-- where 수치형
where vacationhour = 10
where vacationhour != 10
where vacationhour =>10 and vacationhour <20
where vacationhour between 10 and 19 -- 경계포함

-- 두개의 where 조건
-- and, or
-- where절 에서는 and(교집합) 가 or(합집합) 보다 좋음 /성능면(casebycase)


# 'in', 'not in' filter
select * from employee
where managerid between 16 and 21

-- 16&21번만
select * from employee
where managerid in (16,21)
-- where managerid not in (16,21) ; # 289 = 34 + 255


# null
select * from product
where size is null -- not char
-- where size is not null ;


# 정렬 order by
select * from employee
order by gender, hiredate desc ;

select * from employee
order by 9, 10 desc ; -- cols num / 임시로, 개발중에 나오는 결과를 확인하고 싶을 때 (실제사용X)


# quiz : 
-- employeePayhistory 테이블에서 시급(rate)이 높은 순서대로 20명만 출력!!
select * from employeePayhistory
order by rate desc
limit 0,20;


# alias(별칭)
select employeeid as 사원번호
from employee 사원테이블 #??


# 집계함수
count(), sum(), max(), min(), avg()
distinct

select count(*) from salesorderheader -- 31465

select * from salesorderheader -- 31465
where 1=0 -- columns만 알고 싶을 때 # ?? 1=2도 되는데..뭐가 기준 / where 1=1 : 참 (and) / where 1=0 : or

select count(*) as 총주문건수
, max(orderdate) as 최근주문일, min(orderdate) as 최초주문일
, avg(totaldue) 객단가
from salesorderheader

select count(distinct title) from employee; 

select distinct title from employee;

select count(*) from salesorderheader; -- 총 주문건수 # 31465 -> 1row

select TerritoryID, count(*) 
from salesorderheader
group by TerritoryID ; -- 지역별 총 주문건수
-- order by 총주문건수 desc ;

select TerritoryID
from salesorderheader


select * from salesorderheader

select salesPersonID, TerritoryID, count(*) as 총주문건수
from salesorderheader
where salesPersonID is not null
group by salesPersonID, TerritoryID 
order by salesPersonID, TerritoryID ;


select salesPersonID, TerritoryID, count(*) as 총주문건수
from salesorderheader
where salesPersonID is not null
group by salesPersonID, TerritoryID 
having count(*) >=10 -- 집계되어진 값에서 사용하는 것
order by salesPersonID, TerritoryID ;


select * from salesOrderDetail;
-- salesOrderId당 5개 이상의 건수가 있는 주문번호(SalesOrderID)를 조회하는 쿼리는?
select salesorderid, count(*) as 주문건수 -- count(SalesOrderDetailID)
from salesOrderDetail
group by salesorderid
having 주문건수 >=5; 


-- salesorderdetail에서 각 물건당 2개 이상 구매한 건수가 
-- salesOrderId당 5개 이상의 건수가 있는 주문번호(SalesOrderID)를 조회하는 쿼리는?
select salesorderid, count(*) as 주문건수
from salesOrderDetail
where orderqty >=2
group by salesorderid
having 주문건수 >=5
order by 2 desc; 
-- order by 주문건수 desc ;
-- order by salesPersonID, TerritoryID  ;



# join : 두개의 테이블을 '컬럼'의 개수가 늘어나는 방향으로 붙이는 작업
-- 관계형 데이터베이스에서 키를 기반으로 붙이는 것

-- '레코드'가 늘어나는 방향으로 붙이는 방법 -> union, union all
-- 공공데이터가 일별, 월별, 연도별로 파일이 나눠져 있는 경우

-- inner, outer, full join
-- inner join -> 교집합, 두개의 테이블에서 모두 있는 경우 조회
-- (left or right)outer joun -> 해당 위치의 테이블은 모두 가져오고 반대편 테이블 중 없는 것은 null로 표현


-- join 첫 번째, SalesOrderHeader와 SalesOrderDetail 에서 
-- salesOrderHeader의 subtotal , taxamt, freight, totalamt, 
-- salesOrderDetail에서 Orderqty*unitPrice

select * from salesorderheader
select * from salesOrderDetail

select *
from salesOrderheader inner join salesOrderDetail
on salesOrderHeader.Salesorderid = SalesOrderDetail.SalesOrderid
limit 0,30

-- 별칭을 써서 표현
select *
from salesOrderheader Soh inner join salesOrderDetail Sod
on Soh.Salesorderid = Sod.SalesOrderid
limit 0,30

select salesorderid, salesorderdetailid, subtotal , taxamt, freight, totaldue, Orderqty*unitPrice
from salesOrderheader Soh inner join salesOrderDetail Sod
on Soh.Salesorderid = Sod.SalesOrderid
limit 0,30 -- error : ambiguous

select Soh.salesorderid, Sod.salesorderid, salesorderdetailid,  subtotal , taxamt, freight, totaldue, Orderqty*unitPrice 
from salesOrderheader Soh inner join salesOrderDetail Sod
on Soh.Salesorderid = Sod.SalesOrderid
limit 0,30

-- outer join, 방향성이 있다 
-- salesOrderdetail 즉, 주문된 적이 없는 제품 코드는?

select salesorderdetail.productid, product.productid, product.name
from salesorderdetail     join    product 
on salesorderdetail.productid  = product.productid

select sod.productid, p.productid, p.name
from salesorderdetail sod    join    product p
on sod.productid  = p.productid
-- 121317 rows

select sod.productid, p.productid, p.name
from salesorderdetail sod  right outer join    product p
on sod.productid  = p.productid
order by sod.productid
-- 121555 rows

-- 제품은 있지만 한개도 팔리지 않은 제품 리스트 찾기
select sod.productid, p.productid, p.name
from salesorderdetail sod    right outer join    product p
on sod.productid  = p.productid
where sod.productid is null -- 238 rows   # sod.productid  = p.productid인데 어떻게 null 값이 생김?
order by sod.productid

-- 3개 이상의 테이블 조인하기
select P.*
from Product p JOIN ProductSubcategory PS
ON p.ProductSubCategoryID = PS.ProductSubCategoryID
JOIN ProductCategory PC
ON ps.ProductCategoryid = PC.ProductCategoryid

select p.productid, p.name, pc.name, ps.name
from Product p JOIN ProductSubcategory PS
ON p.ProductSubCategoryID = PS.ProductSubCategoryID
JOIN ProductCategory PC
ON ps.ProductCategoryid = PC.ProductCategoryid

-- inner join 하는 테이블의 순서는 상관x
select p.productid, p.name, pc.name, ps.name
from ProductCategory PC JOIN ProductSubcategory PS
ON PC.ProductCategoryID = PS.ProductCategoryID 
JOIN Product P 
ON ps.ProductSubCategoryID = p.ProductSubCategoryID

-- 퀴즈
select soh.salesorderid, sor.salesreasonid, sr.reasontype
from salesorderheader soh join salesorderheadersalesreason sor
on soh.salesorderid = sor.salesorderid
join salesreason sr
on sor.salesreasonid = sr.salesreasonid
group by sr.reasontype

select  sr.reasontype, sr.name,  count(*) 
from salesorderheader soh join salesorderheadersalesreason sor
on soh.salesorderid = sor.salesorderid
join salesreason sr
on sor.salesreasonid = sr.salesreasonid
group by  sr.reasontype, sr.name
having count(*) >=2000


-- view 를 만듦
create view vw_salesreaseon
as
select  sr.reasontype, sr.name,  count(*) as 건수
from salesorderheader soh join salesorderheadersalesreason sor
on soh.salesorderid = sor.salesorderid
join salesreason sr
on sor.salesreasonid = sr.salesreasonid
group by  sr.reasontype, sr.name
;

select * from vw_salesreaseon 
where 건수>100



# union and union all
-- union은 레코드가 증가하는 방향으로 병합
select * from salesorderheader limit 0,5

select salesorderid, orderdate, year(orderdate), SubTotal, TaxAmt, duedate
from salesorderheader
order by orderdate

-- e commerce , iot- > 시계열 데이터가 많다. 
-- 데이터가 너무 많으면 쪼갠다. --> 파티셔닝 (partition)
-- 주로 연 단위로 나눔. 
-- salesorderheader2001, salesorderheader2002, 

create view vw_주문테이블
as
select * from 
(
	select salesorderid, orderdate, year(orderdate), SubTotal, TaxAmt, duedate
	from salesorderheader
	where year(orderdate) =2001
	union all
	select salesorderid, orderdate, year(orderdate), SubTotal, TaxAmt, duedate
	from salesorderheader
	where year(orderdate) =2002
	union all 
	select salesorderid, orderdate, year(orderdate), SubTotal, TaxAmt, duedate
	from salesorderheader
	where year(orderdate) =2003
	union all 
	select salesorderid, orderdate, year(orderdate), SubTotal, TaxAmt, duedate
	from salesorderheader
	where year(orderdate) =2004
) a

union 과 union all의 차이는 전자는 distinct포함의미 , 후자는 모두 나온다. 

select * from vw_주문테이블