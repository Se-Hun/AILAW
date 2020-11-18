import React, { Component } from 'react';
import {Input, Button} from "reactstrap";

class RealPrediction extends Component {
    constructor(props) {
        super(props);
        this.state = {
            taggedData : ""
        };
    };

    _callApi = () =>{
        const url = "http://localhost:5000/extractCrime"
        let formData  = new FormData()
        formData.append("name", this.state.name)

        return fetch(url, {
          method: 'POST',
          body: formData
        }).then( res => res.json())
        .then(data => {
            return data
        })
        .catch(err => console.log(err))
    }

    // _requestPost = async() => {
    //     const receivedData = await this._callApi()
    //
    //     const newState = {
    //         "taggedData" : receivedData["taggedData"]
    //     }
    // }

    _handleChange = (e) => {
      this.setState({
        "name" : e.target.value
      })
      console.log(this.state)
    }

    _submitInput = async(e) => {
        const receivedData = await this._callApi()

        const newState = {
            "taggedData" : receivedData["taggedData"]
        }
    }

    render() {
        return(
            <div>
                <Input type="textarea" name="textarea-input" id="textarea-input" rows="9" placeholder="Content..."
                       id={this.props.id}
                       value={this.props.content} // props변경이 되면 value가 자동으로 바뀜
                       onChange={this._handleChange}/>
                <Button onClick={e => this._submitInput(e)}>Submit</Button>
            </div>
        )
    }
}

export default RealPrediction;