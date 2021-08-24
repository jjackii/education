import React from 'react';

// const friends = [
//     {
//         id: 1,
//         name: '친구1',
//         image: 'http://aeriskitchen.com/wp-content/uploads/2008/09/kimchi_bokkeumbap_02-.jpg',
//     },
//     {
//         id: 2,
//         name: '친구2',
//         image:
//             'https://3.bp.blogspot.com/-hKwIBxIVcQw/WfsewX3fhJI/AAAAAAAAALk/yHxnxFXcfx4ZKSfHS_RQNKjw3bAC03AnACLcBGAs/s400/DSC07624.jpg',
//     },
// ];

// function friendname(a) {
// return a.name;
// }

// console.log( friends.map(friendname) )



function Friend(props) {
    return (

    <div>
        <h1>{props.name}</h1> {/* 속성 전달 */}
        <h1>{props.age}</h1>
        <h1>{props.image}</h1>
    </div>
    )
}

function Image() {
    return (
        <div>
            <Friend name="HOGWARTS" age="1996.03.07" />
            <img src="https://www.freepnglogos.com/uploads/hogwarts-logo-png/hogwarts-logo-hd-picture-21.png" />
        </div>
    )
}

export default Image;