// 1. Feature engineering - process of using domain knowledge to extract features from raw data via data mining techniques.
// 2. Data Collection & Data Verificaton - at least we want to know what data did we get and what our model marked as spam.
// 3. Model creation - combine data and ideas.
// 4. Serving infrastructure - how to perform inference?

package main

import (
	"fmt"
	"os"
	"time"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type User struct {
	Name      string
	Email     string
	CreatedAt time.Time
}

type Message struct {
	User      User
	CreatedAt time.Time
	Text      string
}

func main() {
	oneMinAgo := time.Now().Add(-1 * time.Minute)
	twoMinAgo := time.Now().Add(-2 * time.Minute)

	msg := Message{
		User:      User{Name: "John", Email: "10minutemail.com", CreatedAt: twoMinAgo},
		CreatedAt: oneMinAgo,
		Text:      "You have WON a guaranteed 1000 cash! To get your money, click the link http://russianroulette.com?n=QJKGIGHJJGCBL",
	}

	// Use saved_model_cli to analize model signature
	// docker run -it --rm --net=host -v $(pwd):/workdir -w /workdir tensorflow/tensorflow saved_model_cli show --all --dir model

	dir := "./model"
	model, err := loadSavedModel(dir)
	if err != nil {
		panic(err)
	}

	msgCreatedAtOutput := tf.Output{
		Op:    model.Graph.Operation("signature_msg_created_at"),
		Index: 0,
	}
	msgCreatedAtTensor, _ := tf.NewTensor(int64(msg.CreatedAt.Unix()))

	msgTextOutput := tf.Output{
		Op:    model.Graph.Operation("signature_msg_text"),
		Index: 0,
	}
	msgTextTensor, _ := tf.NewTensor(msg.Text)

	userCreatedAtOutput := tf.Output{
		Op:    model.Graph.Operation("signature_user_created_at"),
		Index: 0,
	}
	userCreatedAtTensor, _ := tf.NewTensor(msg.User.CreatedAt.Unix())

	userEmailOutput := tf.Output{
		Op:    model.Graph.Operation("signature_user_email"),
		Index: 0,
	}
	userEmailTensor, _ := tf.NewTensor(msg.User.Email)

	confidenceOutput := tf.Output{
		Op:    model.Graph.Operation("StatefulPartitionedCall"),
		Index: 0,
	}

	// feeds - input tensors for model
	feeds := map[tf.Output]*tf.Tensor{
		msgCreatedAtOutput:  msgCreatedAtTensor,
		msgTextOutput:       msgTextTensor,
		userCreatedAtOutput: userCreatedAtTensor,
		userEmailOutput:     userEmailTensor,
	}

	// fetches define what model will return for us
	fetches := []tf.Output{confidenceOutput}

	tensors, err := model.Session.Run(feeds, fetches, nil)
	if err != nil {
		panic(err)
	}

	confidence := tensors[0].Value().(float32)

	fmt.Printf("Spam confidence: %f", confidence)
}

func loadSavedModel(dir string) (*tf.SavedModel, error) {
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		return nil, fmt.Errorf("directory \"%s\" doesn't exist", dir)
	}

	metaGraph := "serve"

	model, err := tf.LoadSavedModel(dir, []string{metaGraph}, &tf.SessionOptions{})
	if err != nil {
		return nil, err
	}

	return model, nil
}
